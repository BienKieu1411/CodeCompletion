from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from graphcoder_rl.data.repository_dataset_loader import DatasetLoader
from graphcoder_rl.retrieval.coarse_dense_retriever import CoarseDenseRetriever

logger = logging.getLogger(__name__)


def _collect_samples(dataset_loader: DatasetLoader, max_samples: int, batch_size: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for batch in dataset_loader.get_epoch_batches(batch_size=batch_size):
        for sample in batch:
            samples.append(sample)
            if len(samples) >= max_samples:
                return samples
    return samples


def run_contrastive_pretrain(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = config or {}
    dataset_path = cfg.get("dataset_path", "data/github_repos/python/train.parquet")
    output_dir = cfg.get("output_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg.get("model_name", "microsoft/unixcoder-base")
    batch_size = int(cfg.get("batch_size", 8))
    num_epochs = int(cfg.get("num_epochs", 1))
    max_samples = int(cfg.get("max_samples", 256))
    temperature = float(cfg.get("temperature", 0.07))
    lr = float(cfg.get("learning_rate", 2e-5))

    loader = DatasetLoader(
        dataset_path=dataset_path,
        use_fim=False,
        completion_level=cfg.get("completion_level", "line"),
        fixed_train=True,
        fixed_train_size=max_samples,
    )
    loader.prepare_dataset()
    samples = _collect_samples(loader, max_samples=max_samples, batch_size=batch_size)
    if not samples:
        raise RuntimeError("No training samples available for contrastive pretrain.")

    retriever = CoarseDenseRetriever(model_name=model_name, device=device)
    retriever.train()
    optimizer = torch.optim.AdamW(retriever.model.parameters(), lr=lr)

    global_step = 0
    epoch_losses: List[float] = []
    for epoch in range(num_epochs):
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            if len(batch) < 2:
                continue
            queries = [x["left_context"][-1500:] for x in batch]
            positives = [x["ground_truth"][:1500] for x in batch]

            q_emb = retriever.encode_batch(queries, batch_size=len(queries))
            p_emb = retriever.encode_batch(positives, batch_size=len(positives))
            logits = torch.matmul(q_emb, p_emb.T) / max(temperature, 1e-4)
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retriever.model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            global_step += 1
            if global_step % 20 == 0:
                logger.info("contrastive-pretrain step=%d loss=%.4f", global_step, loss.item())

    ckpt_path = os.path.join(output_dir, "contrastive_pretrain.pt")
    torch.save(
        {
            "coarse_retriever": retriever.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": num_epochs,
            "meta": {
                "framework": "GraphCoderRL",
                "stage": "contrastive_pretrain",
                "samples": len(samples),
            },
        },
        ckpt_path,
    )
    mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
    return {"checkpoint": ckpt_path, "loss": mean_loss, "steps": global_step, "samples": len(samples)}
