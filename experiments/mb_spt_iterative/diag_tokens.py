#!/usr/bin/env python3
"""Verify that passing {'prompt_token_ids': prompt_ids} to vLLM preserves spaces and newlines 100%."""

from __future__ import annotations

from typing import Any


def run_iteration(global_ns: dict[str, Any], subset_per_benchmark: int = 1, tag: str = "diag2") -> dict[str, Any]:
    from vllm import SamplingParams
    engine = global_ns["VLLM_ENGINE"]
    data = global_ns["MB_DATA"]
    tok = data["tokenizer"]
    t = data["tasks"][0]

    prompt_ids = tok.encode(t["left_context"], add_special_tokens=False)[-256:]
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, n=1, seed=123)
    out = engine.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)[0]

    vllm_p_ids = list(out.prompt_token_ids)
    vllm_o_ids = list(out.outputs[0].token_ids)
    vllm_text = out.outputs[0].text
    hf_dec_o = tok.decode(vllm_o_ids, skip_special_tokens=True)
    hf_tokens_o = tok.convert_ids_to_tokens(vllm_o_ids)

    res = {
        "tag": tag,
        "orig_prompt_ids_tail": prompt_ids[-15:],
        "vllm_prompt_ids_tail": vllm_p_ids[-15:],
        "vllm_output_ids": vllm_o_ids,
        "vllm_output_tokens": hf_tokens_o,
        "vllm_text": repr(vllm_text),
        "hf_decode_output": repr(hf_dec_o),
        "overall_by_arm": {},
        "paired_bootstrap_vs_aligncoder_code": {},
    }
    from pathlib import Path
    import json
    out_dir = Path("/content/mb_spt/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    return res
