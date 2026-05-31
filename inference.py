"""
GraphFRL Inference / Evaluation Script (Production)

Pipeline:
1. Load trained retriever checkpoint
2. Run retrieval + generation on test set
3. Save predictions as JSONL
4. Auto-evaluate with EM/ES metrics
"""

import os
import sys
import json
import argparse
import logging
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GraphFRL.retriever.unixcoder_retriever import UniXCoderRetriever
from GraphFRL.retriever.graph_retriever import GraphRetriever
from GraphFRL.data.dataset_loader import GraphFRLDataLoader
from GraphFRL.data.ast_parser import ASTQueryExtractor
from GraphFRL.generator.causal_prompt import CausalPromptGenerator
from GraphFRL.generator.deepseek_generator import DeepSeekGenerator
from GraphFRL.evaluate import evaluate_predictions, save_results

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("GraphFRL.inference")


def infer(args):
    print("\n" + "=" * 60)
    print("=== GRAPHFRL INFERENCE ===")
    print(f"  Model:      {args.model_name}")
    print(f"  Checkpoint: {args.checkpoint or 'None (zero-shot)'}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Initialize Pipeline ───────────────────────────────────────
    logger.info("Loading models...")
    dense_retriever = UniXCoderRetriever(device=device)
    graph_retriever = GraphRetriever()
    data_loader = GraphFRLDataLoader(use_fim=False, completion_level="line")
    ast_extractor = ASTQueryExtractor()
    prompt_gen = CausalPromptGenerator(model_name=args.model_name)
    llm = DeepSeekGenerator(model_name=args.model_name, device=device)

    # ── Load checkpoint ───────────────────────────────────────────
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "retriever" in ckpt:
            dense_retriever.model.load_state_dict(ckpt["retriever"])
        else:
            dense_retriever.model.load_state_dict(ckpt)
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
    start_time = time.time()

    with torch.no_grad():
        for idx, sample in enumerate(test_samples):
            query = sample["left_context"]
            file_path = sample["id"]

            # Graph retrieval
            local_graph = ast_extractor.extract_local_graph(
                query, cursor_line=len(query.split("\n"))
            )
            graph_snippets = graph_retriever.retrieve_paths(
                local_graph=local_graph,
                crossfile_dict=sample["crossfile_context"],
                current_file=file_path,
            )

            # Dense retrieval
            dense_snippets, _ = dense_retriever.retrieve_top_k(
                query, sample["crossfile_context"], top_k=args.top_k
            )

            # Merge context and generate
            repo_snippet = "\n".join(graph_snippets) + "\n\n" + "\n".join(dense_snippets)
            prompt = prompt_gen.construct_prompt(repo_snippet, query, file_path=file_path)

            pred_text, _, _ = llm.generate_with_attention(
                prompt, retrieved_tokens_len=0, max_new_tokens=args.max_gen_length
            )

            predictions.append({
                "task_id": sample["id"],
                "pred": pred_text.strip(),
                "target": sample["ground_truth"],
            })

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed
                logger.info(f"  Progress: {idx + 1}/{len(test_samples)} ({speed:.1f} samples/sec)")

    elapsed = time.time() - start_time
    logger.info(f"\nInference completed in {elapsed:.1f}s ({len(predictions)/elapsed:.1f} samples/sec)")

    # ── Evaluate & Save ───────────────────────────────────────────
    results = save_results(predictions, args.output_dir, args.language)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphFRL Inference")
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
    args = parser.parse_args()
    infer(args)
