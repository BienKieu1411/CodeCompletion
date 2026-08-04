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


def _backend_token_ids(tokenizer: AutoTokenizer, text: str) -> List[int]:
    """Tokenize without HF's misleading overlength warning.

    We intentionally inspect the full token sequence before applying our own
    left-context/target budget. Fast tokenizers expose the Rust backend for
    this operation and do not emit the warning that is triggered by calling
    ``encode`` without a truncation policy.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        try:
            return list(backend.encode(text, add_special_tokens=False).ids)
        except (AttributeError, TypeError, ValueError):
            pass
    return list(tokenizer.encode(text, add_special_tokens=False))


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
        lc_ids = _backend_token_ids(tokenizer, left_context)
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
            text_ids = _backend_token_ids(tokenizer, text)
            n = len(text_ids)
            if used + n > budget_for_chunks:
                remaining = budget_for_chunks - used
                if remaining > 20:
                    trunc_ids = text_ids[:remaining]
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
        Any HuggingFace causal LM, e.g.
        ``deepseek-ai/deepseek-coder-6.7b-base``.
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
        model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
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
        self.model.requires_grad_(False)
        # The generator is a teacher/inference model. Only the prompt tensor
        # below may receive gradients; disabling KV caching also avoids
        # retaining useless cache tensors during prompt warm-up/NLL scoring.
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

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

    def assert_backbone_frozen(self) -> None:
        """Fail fast if any generator backbone parameter became trainable."""
        trainable = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            preview = ", ".join(trainable[:8])
            raise RuntimeError(
                "Generator backbone must remain frozen; trainable parameters: "
                f"{preview}"
            )
        if self.model.training:
            self.model.eval()

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
        self,
        text: str,
        *,
        max_text_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenise *text* and prepend soft prompt embeddings.

        Returns (inputs_embeds, attention_mask).
        """
        token_budget = (
            max_text_tokens
            if max_text_tokens is not None
            else self.budget_manager.max_tokens - self.num_prompt_tokens
        )
        token_budget = max(1, token_budget)
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=token_budget,
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

    def _prepare_teacher_forcing_ids(
        self,
        context_text: str,
        target: str,
        *,
        use_soft_prompt: bool,
    ) -> Tuple[List[int], List[int], int]:
        """Build bounded context/target ids without invoking model compute."""
        context_ids = _backend_token_ids(self.tokenizer, context_text)
        target_ids = _backend_token_ids(self.tokenizer, target)
        prompt_offset = self.num_prompt_tokens if use_soft_prompt else 0
        text_budget = self.budget_manager.max_tokens - prompt_offset
        if text_budget <= 0:
            return [], [], prompt_offset

        if len(target_ids) >= text_budget:
            target_ids = target_ids[:text_budget]
            context_ids = []
        else:
            context_budget = text_budget - len(target_ids)
            context_ids = (
                context_ids[-context_budget:] if context_budget > 0 else []
            )
        return context_ids, target_ids, prompt_offset

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
        use_soft_prompt : bool — whether to apply the generator adapter.
            C_stop uses the same value as retrieved candidates; it differs only
            by having no retrieved chunks.

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

        context_ids, target_ids, prompt_offset = self._prepare_teacher_forcing_ids(
            context_text,
            target,
            use_soft_prompt=use_soft_prompt,
        )
        if not target_ids:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

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

    @torch.inference_mode()
    def teacher_forcing_nll_batch(
        self,
        left_context: str,
        target: str,
        retrieved_chunks_list: Sequence[Optional[Sequence[CodeChunk]]],
        use_soft_prompt: bool = True,
        micro_batch_size: int = 2,
    ) -> List[float]:
        """Score several context candidates in one frozen-model forward.

        Phase 2 compares multiple retrieval strategies for the same example.
        Batching those candidates preserves exact token-level NLL while
        replacing one model launch per strategy with one padded batch.  The
        candidate list is split into small micro-batches so the full-vocab
        logits do not multiply the generator's peak VRAM by the number of
        retrieval strategies (usually 5--7).
        """
        if not retrieved_chunks_list:
            return []
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if len(retrieved_chunks_list) > micro_batch_size:
            values: List[float] = []
            for start in range(0, len(retrieved_chunks_list), micro_batch_size):
                values.extend(
                    self.teacher_forcing_nll_batch(
                        left_context,
                        target,
                        retrieved_chunks_list[start : start + micro_batch_size],
                        use_soft_prompt=use_soft_prompt,
                        micro_batch_size=micro_batch_size,
                    )
                )
            return values

        rows: List[Tuple[List[int], List[int], int]] = []
        for retrieved_chunks in retrieved_chunks_list:
            if retrieved_chunks:
                context_text = self.budget_manager.pack(
                    retrieved_chunks, left_context, self.tokenizer
                )
            else:
                context_text = left_context
            rows.append(
                self._prepare_teacher_forcing_ids(
                    context_text,
                    target,
                    use_soft_prompt=use_soft_prompt,
                )
            )

        valid_rows = [index for index, (_, target_ids, _) in enumerate(rows) if target_ids]
        if not valid_rows:
            return [0.0 for _ in rows]

        prompt_offset = rows[valid_rows[0]][2]
        max_text_len = max(
            len(context_ids) + len(target_ids)
            for context_ids, target_ids, _ in rows
        )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        input_ids = torch.full(
            (len(rows), max_text_len),
            int(pad_id),
            dtype=torch.long,
            device=self._device,
        )
        text_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)

        for row_index, (context_ids, target_ids, row_prompt_offset) in enumerate(rows):
            if not target_ids:
                continue
            ids = context_ids + target_ids
            length = len(ids)
            input_ids[row_index, :length] = torch.tensor(
                ids, dtype=torch.long, device=self._device
            )
            text_mask[row_index, :length] = 1
            target_start = len(context_ids)
            labels[row_index, target_start:length] = torch.tensor(
                target_ids, dtype=torch.long, device=self._device
            )

        with torch.no_grad():
            text_embeds = self.model.get_input_embeddings()(input_ids)
        if use_soft_prompt:
            prompt_embeds = self.prompt_embeddings.unsqueeze(0).to(
                dtype=text_embeds.dtype, device=self._device
            ).expand(len(rows), -1, -1)
            inputs_embeds = torch.cat([prompt_embeds, text_embeds], dim=1)
            prompt_mask = torch.ones(
                len(rows), self.num_prompt_tokens,
                dtype=torch.long,
                device=self._device,
            )
            attention_mask = torch.cat([prompt_mask, text_mask], dim=1)
            padded_labels = torch.cat(
                [
                    torch.full(
                        (len(rows), self.num_prompt_tokens),
                        -100,
                        dtype=torch.long,
                        device=self._device,
                    ),
                    labels,
                ],
                dim=1,
            )
        else:
            inputs_embeds = text_embeds
            attention_mask = text_mask
            padded_labels = labels

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :].float()
        shifted_labels = padded_labels[:, 1:]
        token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shifted_labels.shape)
        token_mask = shifted_labels.ne(-100)
        totals = (token_loss * token_mask).sum(dim=1)
        counts = token_mask.sum(dim=1).clamp_min(1)
        values = (totals / counts).detach().cpu().tolist()
        return [float(value) if target_ids else 0.0 for value, (_, target_ids, _) in zip(values, rows)]

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
    def _greedy_with_cache(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> Optional[List[int]]:
        """Greedy decode with KV cache; return ``None`` if unsupported."""
        try:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )
        except TypeError:
            return None
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            return None

        generated_ids: List[int] = []
        next_id = outputs.logits[0, -1, :].argmax()
        for _ in range(max_new_tokens):
            if next_id.item() == self.tokenizer.eos_token_id:
                break
            generated_ids.append(next_id.item())
            next_embed = self.model.get_input_embeddings()(
                next_id.view(1, 1)
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(1, 1, dtype=torch.long, device=self._device),
                ],
                dim=1,
            )
            outputs = self.model(
                inputs_embeds=next_embed,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                return None
            next_id = outputs.logits[0, -1, :].argmax()
        return generated_ids

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
            inputs_embeds, attention_mask = self._prepare_inputs(
                context_text,
                max_text_tokens=(
                    self.budget_manager.max_tokens
                    - self.num_prompt_tokens
                    - max_new_tokens
                ),
            )
        else:
            tokens = self.tokenizer(
                context_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.budget_manager.max_tokens - max_new_tokens,
            ).to(self._device)
            inputs_embeds = self.model.get_input_embeddings()(tokens.input_ids)
            attention_mask = tokens.attention_mask

        cached_ids = self._greedy_with_cache(
            inputs_embeds,
            attention_mask,
            max_new_tokens,
        )
        if cached_ids is not None:
            return self.tokenizer.decode(cached_ids, skip_special_tokens=True)

        # Compatibility fallback for custom causal-LM implementations that do
        # not expose past_key_values. This path is slower but preserves the
        # previous behavior.
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

    @torch.no_grad()
    def generate_drafts(
        self,
        left_context: str,
        num_drafts: int = 2,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.95,
        use_soft_prompt: bool = True,
    ) -> List[str]:
        """Sample lightweight no-retrieval drafts for query enhancement."""
        if num_drafts < 1 or max_new_tokens < 1:
            return []
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

        drafts: List[str] = []
        for _ in range(num_drafts):
            if use_soft_prompt:
                inputs_embeds, attention_mask = self._prepare_inputs(
                    left_context,
                    max_text_tokens=(
                        self.budget_manager.max_tokens
                        - self.num_prompt_tokens
                        - max_new_tokens
                    ),
                )
            else:
                tokens = self.tokenizer(
                    left_context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.budget_manager.max_tokens - max_new_tokens,
                ).to(self._device)
                inputs_embeds = self.model.get_input_embeddings()(tokens.input_ids)
                attention_mask = tokens.attention_mask

            generated_ids: List[int] = []
            for _step in range(max_new_tokens):
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_id = sorted_indices[sampled].squeeze(0)
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
            drafts.append(
                self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
        return drafts

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
