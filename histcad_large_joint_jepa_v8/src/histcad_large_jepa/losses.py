from __future__ import annotations

import torch
from torch.nn import functional as F


def cosine_prediction(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (2.0 - 2.0 * (prediction * target).sum(-1)).mean()


def symmetric_info_nce(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = left @ right.T / temperature
    labels = torch.arange(len(left), device=left.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def redundancy_regularization(
    raw: torch.Tensor, permutation: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(raw) < 2:
        zero = raw.sum() * 0.0
        return zero, zero
    centered = raw - raw.mean(0)
    standard_deviation = torch.sqrt(centered.var(0) + 1e-4)
    variance = F.relu(0.5 - standard_deviation).mean()
    standardized = centered / standard_deviation.clamp_min(1e-4)
    standardized = standardized[:, permutation]
    covariance_terms = []
    for start in range(0, standardized.shape[1], block_size):
        block = standardized[:, start : start + block_size]
        covariance = block.T @ block / (len(block) - 1)
        off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
        covariance_terms.append(off_diagonal.square().mean())
    return variance, torch.stack(covariance_terms).mean()


def near_duplicate_loss(embedding: torch.Tensor, maximum_cosine: float = 0.95) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    similarities = embedding @ embedding.T
    off_diagonal = ~torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    return F.relu(similarities[off_diagonal] - maximum_cosine).square().mean()


def training_loss(
    outputs: dict[str, torch.Tensor],
    dimension_target: torch.Tensor,
    permutations: dict[str, torch.Tensor],
    temperature: float,
    block_size: int,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    terms = {
        "joint_jepa": cosine_prediction(outputs["joint_prediction"], outputs["joint_target"]),
        "parametric_jepa": cosine_prediction(
            outputs["parametric_prediction"], outputs["parametric_target"]
        ),
        "step_jepa": cosine_prediction(outputs["step_prediction"], outputs["step_target"]),
        "cross_source_alignment": symmetric_info_nce(
            outputs["parametric_alignment"], outputs["step_alignment"], temperature
        ),
        "dimension": F.smooth_l1_loss(outputs["dimension_prediction"], dimension_target),
        "slot_diversity": outputs["slot_diversity"],
        "near_duplicate": near_duplicate_loss(outputs["joint"]),
    }
    joint_variance, joint_covariance = redundancy_regularization(
        outputs["joint_raw"], permutations["joint"], block_size
    )
    parametric_variance, parametric_covariance = redundancy_regularization(
        outputs["parametric_raw"],
        permutations["parametric"],
        min(block_size, outputs["parametric_raw"].shape[1]),
    )
    step_variance, step_covariance = redundancy_regularization(
        outputs["step_raw"],
        permutations["step"],
        min(block_size, outputs["step_raw"].shape[1]),
    )
    terms.update(
        joint_variance=joint_variance,
        joint_covariance=joint_covariance,
        branch_variance=(parametric_variance + step_variance) / 2.0,
        branch_covariance=(parametric_covariance + step_covariance) / 2.0,
    )
    loss = sum(weights.get(name, 0.0) * value for name, value in terms.items())
    return loss, {
        "loss": float(loss.detach()),
        **{name: float(value.detach()) for name, value in terms.items()},
    }
