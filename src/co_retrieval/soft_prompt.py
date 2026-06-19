"""Soft prompt tuning for frozen code LLM generator.

The generator backbone (e.g. Qwen2.5-Coder-7B-Instruct) is **completely
frozen**.  Only the soft prompt embedding matrix is trainable, updated by
generation CE loss.

Key capabilities
----------------
* **Instruction-based initialisation** — prompt embeddings seeded from a
  natural-language instruction.
* **Teacher-forcing NLL** — compute negative log-likelihood on the target
  completion in a single forward pass (no autoregressive decoding).
  This is the signal used to build DPO preference data.
* **Context mixing** — warm-up Phase 1 uses a mixture of no-context (20%),
  oracle context (50%), and noisy retrieved context (30%).
* **Token budget** — packs soft prompt + retrieved snippets + left context
  within the model's context window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

from co_retrieval.chunking import CodeChunk

logger = logging.getLogger(__name__)

# ── Default instruction ───────────────────────────────────────────────────────

DEFAULT_GENERATOR_INSTRUCTION = (
    "Below are relevant code snippets retrieved from the repository. "
    "Use them as context to complete the code at the cursor position accurately."
)


# ── Token budget ──────────────────────────────────────────────────────────────


@dataclass
class TokenBudgetManager:
    """Allocate context window across prompt / retrieval / left-context.

    Priority (descending):
    1. Generation headroom (reserved for target / output).
    2. Soft prompt tokens (fixed).
    3. Left context tail (most important real input — closest to cursor).
    4. Retrieved chunks (fill remaining budget, trim lowest-rank first).
    """

    max_tokens: int = 4096
    num_prompt_tokens: int = 50
    generation_headroom: int = 256
    left_context_max_tokens: int = 1500

    @property
    def retrieval_budget(self) -> int:
        return max(
            0,
            self.max_tokens
            - self.num_prompt_tokens
            - self.left_context_max_tokens
            - self.generation_headroom,
        )

    def pack(
        self,
        retrieved_chunks: Sequence[CodeChunk],
        left_context: str,
        tokenizer: AutoTokenizer,
    ) -> str:
        """Build the text portion of the LLM prompt (without soft prompt).

        Layout: ``[chunk_1]\\n[chunk_2]\\n...\\n[left_context_tail]``
        """
        # Left context: keep tail (closest to cursor)
        lc_ids = tokenizer.encode(left_context, add_special_tokens=False)
        if len(lc_ids) > self.left_context_max_tokens:
            lc_ids = lc_ids[-self.left_context_max_tokens :]
        left_text = tokenizer.decode(lc_ids, skip_special_tokens=True)
        left_token_count = len(lc_ids)

        budget_for_chunks = max(
            0,
            self.max_tokens
            - self.num_prompt_tokens
            - left_token_count
            - self.generation_headroom,
        )

        # Pack chunks (highest rank first)
        chunk_parts: List[str] = []
        used = 0
        for chunk in retrieved_chunks:
            text = chunk.retrieval_text()
            n = len(tokenizer.encode(text, add_special_tokens=False))
            if used + n > budget_for_chunks:
                remaining = budget_for_chunks - used
                if remaining > 20:
                    trunc_ids = tokenizer.encode(
                        text, add_special_tokens=False
                    )[:remaining]
                    chunk_parts.append(
                        tokenizer.decode(trunc_ids, skip_special_tokens=True)
                    )
                break
            chunk_parts.append(text)
            used += n

        if chunk_parts:
            return "\n".join(chunk_parts) + "\n" + left_text
        return left_text


# ── SoftPromptLLM ─────────────────────────────────────────────────────────────


class SoftPromptLLM(nn.Module):
    """Frozen code LLM with trainable soft prompt embeddings.

    Parameters
    ----------
    model_name : str
        Any HuggingFace causal LM (e.g. ``Qwen/Qwen2.5-Coder-7B-Instruct``).
    num_prompt_tokens : int
        Number of learnable prompt tokens to prepend.
    max_context_tokens : int
        Total context window of the model.
    device, dtype : hardware settings.
    init_instruction : str | None
        Instruction whose embeddings seed the prompt. Defaults to
        ``DEFAULT_GENERATOR_INSTRUCTION``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        num_prompt_tokens: int = 50,
        max_context_tokens: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        init_instruction: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_prompt_tokens = num_prompt_tokens
        self._device = device
        self._dtype = dtype

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Frozen LLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Learnable prompt embeddings
        embed_dim = self.model.get_input_embeddings().weight.shape[1]
        self.prompt_embeddings = nn.Parameter(
            self._init_from_instruction(
                init_instruction or DEFAULT_GENERATOR_INSTRUCTION, embed_dim
            )
        )

        # Token budget manager
        self.budget_manager = TokenBudgetManager(
            max_tokens=max_context_tokens,
            num_prompt_tokens=num_prompt_tokens,
        )

        self.to(device)

    # ── Instruction-based init ────────────────────────────────────────────

    def _init_from_instruction(
        self, instruction: str, embed_dim: int
    ) -> torch.Tensor:
        tokens = self.tokenizer(instruction, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(tokens.input_ids)

        seq_len = embeddings.shape[1]
        if seq_len >= self.num_prompt_tokens:
            init_embeds = embeddings[0, : self.num_prompt_tokens]
        else:
            mean_embed = embeddings[0].mean(dim=0, keepdim=True)
            padding = mean_embed.expand(self.num_prompt_tokens - seq_len, -1)
            init_embeds = torch.cat([embeddings[0], padding], dim=0)

        logger.info(
            "SoftPromptLLM: init %d prompt tokens from instruction "
            '("%s…", dim=%d)',
            self.num_prompt_tokens,
            instruction[:50],
            embed_dim,
        )
        return init_embeds.clone().float()

    # ── Build inputs with soft prompt ─────────────────────────────────────

    def _prepare_inputs(
        self, text: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenise *text* and prepend soft prompt embeddings.

        Returns (inputs_embeds, attention_mask).
        """
        max_text_tokens = (
            self.budget_manager.max_tokens - self.num_prompt_tokens
        )
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_text_tokens,
        ).to(self._device)

        with torch.no_grad():
            text_embeds = self.model.get_input_embeddings()(tokens.input_ids)

        prompt_embeds = self.prompt_embeddings.unsqueeze(0).to(
            dtype=text_embeds.dtype, device=self._device
        )
        inputs_embeds = torch.cat([prompt_embeds, text_embeds], dim=1)

        prompt_mask = torch.ones(
            1, self.num_prompt_tokens, dtype=torch.long, device=self._device
        )
        attention_mask = torch.cat([prompt_mask, tokens.attention_mask], dim=1)

        return inputs_embeds, attention_mask

    # ── Teacher-forcing NLL (core method — no decoding) ───────────────────

    def teacher_forcing_nll(
        self,
        left_context: str,
        target: str,
        retrieved_chunks: Optional[Sequence[CodeChunk]] = None,
        use_soft_prompt: bool = True,
    ) -> torch.Tensor:
        """Compute NLL on *target* via teacher forcing (single forward pass).

        This is the signal used to build DPO preference data:
        lower NLL = better context for this completion.

        Parameters
        ----------
        left_context : str
        target : str — ground truth completion.
        retrieved_chunks : optional — if given, pack into prompt.
        use_soft_prompt : bool — False for C_stop (no prompt, no context).

        Returns
        -------
        Scalar NLL tensor (negative log-likelihood, lower = better).
        """
        # Build context text
        if retrieved_chunks:
            context_text = self.budget_manager.pack(
                retrieved_chunks, left_context, self.tokenizer
            )
        else:
            context_text = left_context

        context_ids = self.tokenizer.encode(
            context_text, add_special_tokens=False
        )
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        prompt_offset = self.num_prompt_tokens if use_soft_prompt else 0
        text_budget = self.budget_manager.max_tokens - prompt_offset
        if text_budget <= 0:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        if len(target_ids) >= text_budget:
            target_ids = target_ids[:text_budget]
            context_ids = []
        else:
            context_budget = text_budget - len(target_ids)
            context_ids = (
                context_ids[-context_budget:] if context_budget > 0 else []
            )

        input_ids = torch.tensor(
            [context_ids + target_ids],
            dtype=torch.long,
            device=self._device,
        )

        if use_soft_prompt:
            with torch.no_grad():
                text_embeds = self.model.get_input_embeddings()(input_ids)
            prompt_embeds = self.prompt_embeddings.unsqueeze(0).to(
                dtype=text_embeds.dtype, device=self._device
            )
            inputs_embeds = torch.cat([prompt_embeds, text_embeds], dim=1)
            prompt_mask = torch.ones(
                1, self.num_prompt_tokens, dtype=torch.long, device=self._device
            )
            text_mask = torch.ones_like(
                input_ids, dtype=torch.long, device=self._device
            )
            attention_mask = torch.cat([prompt_mask, text_mask], dim=1)
        else:
            # No soft prompt — raw LLM
            with torch.no_grad():
                inputs_embeds = self.model.get_input_embeddings()(
                    input_ids
                )
            attention_mask = torch.ones_like(
                input_ids, dtype=torch.long, device=self._device
            )

        total_len = inputs_embeds.shape[1]

        labels = torch.full(
            (1, total_len), -100, dtype=torch.long, device=self._device
        )

        # Target token positions
        target_start = prompt_offset + len(context_ids)
        target_end = min(target_start + len(target_ids), total_len)
        for j, pos in enumerate(range(target_start, target_end)):
            labels[0, pos] = target_ids[j]

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        if outputs.loss is None:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        return outputs.loss

    # ── Generation loss (for Phase 1 soft prompt warm-up) ─────────────────

    def generation_loss(
        self,
        left_context: str,
        target: str,
        retrieved_chunks: Optional[Sequence[CodeChunk]] = None,
    ) -> torch.Tensor:
        """Cross-entropy generation loss — gradient flows to prompt_embeddings only.

        Alias for ``teacher_forcing_nll`` with ``use_soft_prompt=True``.
        """
        return self.teacher_forcing_nll(
            left_context, target, retrieved_chunks, use_soft_prompt=True
        )

    # ── Generate (inference / evaluation only) ────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        left_context: str,
        max_new_tokens: int = 128,
        retrieved_chunks: Optional[Sequence[CodeChunk]] = None,
        use_soft_prompt: bool = True,
    ) -> str:
        """Greedy autoregressive generation (for final evaluation)."""
        if retrieved_chunks:
            context_text = self.budget_manager.pack(
                retrieved_chunks, left_context, self.tokenizer
            )
        else:
            context_text = left_context

        if use_soft_prompt:
            inputs_embeds, attention_mask = self._prepare_inputs(context_text)
        else:
            tokens = self.tokenizer(
                context_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.budget_manager.max_tokens - max_new_tokens,
            ).to(self._device)
            inputs_embeds = self.model.get_input_embeddings()(tokens.input_ids)
            attention_mask = tokens.attention_mask

        generated_ids: List[int] = []
        for _ in range(max_new_tokens):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            next_logits = outputs.logits[0, -1, :]
            next_id = next_logits.argmax()

            if next_id.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_id.item())
            next_embed = self.model.get_input_embeddings()(
                next_id.unsqueeze(0).unsqueeze(0)
            )
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(1, 1, dtype=torch.long, device=self._device),
                ],
                dim=1,
            )

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save_prompt(self, path: str) -> None:
        torch.save(
            {
                "prompt_embeddings": self.prompt_embeddings.data,
                "num_prompt_tokens": self.num_prompt_tokens,
                "model_name": self.model_name,
            },
            path,
        )
        logger.info("SoftPromptLLM: saved prompt to %s", path)

    def load_prompt(self, path: str) -> None:
        state = torch.load(path, map_location=self._device)
        self.prompt_embeddings = nn.Parameter(
            state["prompt_embeddings"].to(self._device)
        )
        logger.info("SoftPromptLLM: loaded prompt from %s", path)
