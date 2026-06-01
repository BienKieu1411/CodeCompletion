"""
GraphCoderRL Inference / Evaluation Script (Production)

Pipeline:
1. Load trained retriever checkpoint
2. Run retrieval + generation on test set
3. Save predictions as JSONL
4. Auto-evaluate with EM/ES metrics
"""

import os
import json
import argparse
import logging
import time

import torch

from graphcoder_rl.retrieval.coarse_dense_retriever import CoarseDenseRetriever
from graphcoder_rl.retrieval.multi_hop_graph_retriever import MultiHopGraphRetriever
from graphcoder_rl.data.repository_dataset_loader import DatasetLoader
from graphcoder_rl.data.left_context_anchor_extractor import LeftContextAnchorExtractor
from graphcoder_rl.generation.graphcoder_prompt_builder import GraphCoderPromptBuilder
from graphcoder_rl.generation.code_llm_generator import CodeLLMGenerator
from graphcoder_rl.rl.graph_traversal_policy import GraphTraversalPolicy
from graphcoder_rl.evaluation.graphcoder_rl_eval import evaluate_predictions, save_results

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("GraphCoderRL.inference")


def infer(args):
    print("\n" + "=" * 60)
    print("=== GRAPHCODERRL INFERENCE ===")
    print(f"  Model:      {args.model_name}")
    print(f"  Checkpoint: {args.checkpoint or 'None (zero-shot)'}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Initialize Pipeline ───────────────────────────────────────
    logger.info("Loading models...")
    dense_retriever = CoarseDenseRetriever(device=device)
    graph_retriever = MultiHopGraphRetriever(
        enable_left_context_anchors=bool(getattr(args, "enable_left_context_anchors", True)),
        enable_quantization=bool(getattr(args, "enable_quantization", True)),
        enable_multi_hop=bool(getattr(args, "enable_multi_hop", True)),
        enable_structural_edges=bool(getattr(args, "enable_structural_edges", True)),
        enable_control_dependency=bool(getattr(args, "enable_control_dependency", True)),
        enable_override_edges=bool(getattr(args, "enable_override_edges", True)),
        use_graph_cache=bool(getattr(args, "use_graph_cache", True)),
        use_ppl_entropy_cache=bool(getattr(args, "use_ppl_entropy_cache", True)),
    )
    data_loader = DatasetLoader(use_fim=False, completion_level="line")
    ast_extractor = LeftContextAnchorExtractor()
    prompt_gen = GraphCoderPromptBuilder(model_name=args.model_name)
    llm = CodeLLMGenerator(model_name=args.model_name, device=device)
    graph_policy = GraphTraversalPolicy(input_dim=8).to(device)
    use_graph_policy = False

    # ── Load checkpoint ───────────────────────────────────────────
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "coarse_retriever" in ckpt:
            dense_retriever.model.load_state_dict(ckpt["coarse_retriever"])
        elif "retriever" in ckpt:  # legacy fallback
            dense_retriever.model.load_state_dict(ckpt["retriever"])
        else:
            dense_retriever.model.load_state_dict(ckpt)
        if "graph_traversal_policy" in ckpt:
            graph_policy.load_state_dict(ckpt["graph_traversal_policy"])
            graph_policy.eval()
            use_graph_policy = True
        elif "graph_policy" in ckpt:  # legacy fallback
            graph_policy.load_state_dict(ckpt["graph_policy"])
            graph_policy.eval()
            use_graph_policy = True
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logger.warning("No checkpoint found. Running in zero-shot mode.")

    dense_retriever.eval()

    # ── Load test data ────────────────────────────────────────────
    test_samples = data_loader.load_test_samples(
        dataset_name=args.dataset,
        language=args.language,
        max_samples=args.max_samples,
    )

    # ── Inference loop ────────────────────────────────────────────
    logger.info(f"Running inference on {len(test_samples)} samples...\n")
    predictions = []
    retrieval_metas = []
    start_time = time.time()

    with torch.no_grad():
        for idx, sample in enumerate(test_samples):
            query = sample["left_context"]
            file_path = sample["id"]

            # Coarse retrieval first (limits graph traversal space)
            dense_snippets, _, aux = dense_retriever.retrieve_top_k(
                query, sample["crossfile_context"], top_k=args.top_k, return_aux=True
            )

            # Graph retrieval
            local_graph = ast_extractor.extract_local_graph(
                query, cursor_line=len(query.split("\n")), file_path=file_path
            )
            if bool(getattr(args, "save_retrieval_meta", False)):
                graph_snippets, graph_meta = graph_retriever.retrieve_paths(
                    local_graph=local_graph,
                    crossfile_dict=sample["crossfile_context"],
                    current_file=file_path,
                    left_context=query,
                    return_metadata=True,
                    coarse_candidate_chunks=aux.get("filenames", []),
                    policy_model=graph_policy if use_graph_policy else None,
                    policy_device=device,
                )
            else:
                graph_snippets = graph_retriever.retrieve_paths(
                    local_graph=local_graph,
                    crossfile_dict=sample["crossfile_context"],
                    current_file=file_path,
                    left_context=query,
                    coarse_candidate_chunks=aux.get("filenames", []),
                    policy_model=graph_policy if use_graph_policy else None,
                    policy_device=device,
                )
                graph_meta = None

            # Merge context and generate
            repo_snippet = "\n".join(graph_snippets) + "\n\n" + "\n".join(dense_snippets)
            prompt = prompt_gen.construct_prompt(
                repo_snippet, query, file_path=file_path, local_graph=local_graph
            )

            pred_text, _, _ = llm.generate_with_attention(
                prompt, retrieved_tokens_len=0, max_new_tokens=args.max_gen_length
            )

            predictions.append({
                "task_id": sample["id"],
                "pred": pred_text.strip(),
                "target": sample["ground_truth"],
            })
            if graph_meta is not None:
                retrieval_metas.append({
                    "task_id": sample["id"],
                    "metadata": graph_meta,
                })

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed
                logger.info(f"  Progress: {idx + 1}/{len(test_samples)} ({speed:.1f} samples/sec)")

    elapsed = time.time() - start_time
    logger.info(f"\nInference completed in {elapsed:.1f}s ({len(predictions)/elapsed:.1f} samples/sec)")

    # ── Evaluate & Save ───────────────────────────────────────────
    results = save_results(predictions, args.output_dir, args.language)
    if retrieval_metas:
        meta_path = os.path.join(args.output_dir, "retrieval_metadata.jsonl")
        with open(meta_path, "w", encoding="utf-8") as f:
            for item in retrieval_metas:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Add lightweight aggregate metrics for RQ analysis.
        avg_token_cost = sum(float(x["metadata"].get("token_cost", 0.0)) for x in retrieval_metas) / max(1, len(retrieval_metas))
        avg_path_relevance = sum(float(x["metadata"].get("path_relevance", 0.0)) for x in retrieval_metas) / max(1, len(retrieval_metas))
        results["avg_token_cost"] = round(avg_token_cost, 4)
        results["avg_path_relevance"] = round(avg_path_relevance, 4)

    print(f"\n{'=' * 50}")
    print(f"  RESULTS ({args.dataset})")
    print(f"{'=' * 50}")
    print(f"  Exact Match (EM):     {results['em']}%")
    print(f"  Edit Similarity (ES): {results['es']}%")
    print(f"  Identifier EM:        {results['id_em']}%")
    print(f"  Identifier F1:        {results['id_f1']}%")
    print(f"  Total samples:        {results['total']}")
    print(f"{'=' * 50}")

    return results


def build_arg_parser():
    parser = argparse.ArgumentParser(description="GraphCoderRL Inference")
    parser.add_argument("--model_name", type=str,
                        default="deepseek-ai/deepseek-coder-1.3b-base",
                        help="Generator model name/path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to retriever checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="repoeval",
                        help="Test dataset name (repoeval, cceval, etc.)")
    parser.add_argument("--language", type=str, default="python",
                        help="Programming language")
    parser.add_argument("--output_dir", type=str, default="results/inference",
                        help="Output directory for predictions and results")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max test samples (0 = all)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top-k snippets to retrieve")
    parser.add_argument("--max_gen_length", type=int, default=64,
                        help="Max tokens to generate")
    parser.add_argument("--save_retrieval_meta", action="store_true",
                        help="Save per-sample retrieval metadata for research analysis")
    parser.add_argument("--no-left-context-anchors", action="store_true",
                        help="Disable left-context anchors (ablation)")
    parser.add_argument("--no-quantization", action="store_true",
                        help="Disable semantic state quantization (ablation)")
    parser.add_argument("--no-multi-hop", action="store_true",
                        help="Disable multi-hop traversal (ablation)")
    parser.add_argument("--no-structural-edges", action="store_true",
                        help="Disable structural graph edges (ablation)")
    parser.add_argument("--no-control-dependency", action="store_true",
                        help="Disable control_dependency edges (ablation)")
    parser.add_argument("--no-overrides", action="store_true",
                        help="Disable overrides edges (ablation)")
    parser.add_argument("--no-graph-cache", action="store_true",
                        help="Disable repository graph cache")
    parser.add_argument("--no-ppl-cache", action="store_true",
                        help="Disable PPL/entropy cache")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.enable_left_context_anchors = not args.no_left_context_anchors
    args.enable_quantization = not args.no_quantization
    args.enable_multi_hop = not args.no_multi_hop
    args.enable_structural_edges = not args.no_structural_edges
    args.enable_control_dependency = not args.no_control_dependency
    args.enable_override_edges = not args.no_overrides
    args.use_graph_cache = not args.no_graph_cache
    args.use_ppl_entropy_cache = not args.no_ppl_cache
    return infer(args)


if __name__ == "__main__":
    main()
