from __future__ import annotations

import argparse
import glob
import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.speculative.dspark_components.dspark_sts import (
    DSparkStsCalibration,
)

logger = logging.getLogger(__name__)

_EPS_PROB = 1e-8


def default_temperature_grid() -> torch.Tensor:
    return torch.logspace(math.log10(0.1), math.log10(10.0), steps=41)


def expected_calibration_error(
    *,
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int,
) -> float:
    probs = probs.reshape(-1).to(torch.float64).clamp(_EPS_PROB, 1.0 - _EPS_PROB)
    targets = targets.reshape(-1).to(torch.float64)
    total = probs.numel()
    if total == 0:
        return float("nan")
    bin_index = (probs * num_bins).long().clamp_(0, num_bins - 1)
    count = torch.zeros(num_bins, dtype=torch.float64)
    pred_sum = torch.zeros(num_bins, dtype=torch.float64)
    target_sum = torch.zeros(num_bins, dtype=torch.float64)
    count.scatter_add_(0, bin_index, torch.ones_like(probs))
    pred_sum.scatter_add_(0, bin_index, probs)
    target_sum.scatter_add_(0, bin_index, targets)
    denom = count.clamp_min(1.0)
    bin_error = (pred_sum / denom - target_sum / denom).abs()
    return float((bin_error * count).sum().item() / total)


def survival_probabilities(
    *, logits: torch.Tensor, temperatures: Optional[list[float]] = None
) -> torch.Tensor:
    logits = logits.to(torch.float64)
    if temperatures is not None:
        temperature_tensor = torch.tensor(
            temperatures,
            dtype=torch.float64,
            device=logits.device,
        )
        if temperature_tensor.numel() != logits.shape[1]:
            raise ValueError(
                f"STS temperature count {temperature_tensor.numel()} does not match "
                f"logits gamma {logits.shape[1]}."
            )
        if bool((temperature_tensor <= 0).any().item()):
            raise ValueError(f"STS temperatures must all be > 0, got {temperatures}.")
        logits = logits / temperature_tensor.view(1, -1)
    return torch.cumprod(torch.sigmoid(logits), dim=1)


def _brier_scores(*, probs: torch.Tensor, targets: torch.Tensor) -> list[float]:
    return ((probs - targets.to(torch.float64)) ** 2).mean(dim=0).tolist()


def reliability_bins(
    *,
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int,
) -> list[list[dict[str, Optional[float] | int]]]:
    probs = probs.to(torch.float64).clamp(_EPS_PROB, 1.0 - _EPS_PROB)
    targets = targets.to(torch.float64)
    all_bins: list[list[dict[str, Optional[float] | int]]] = []
    for position in range(probs.shape[1]):
        position_probs = probs[:, position].reshape(-1)
        position_targets = targets[:, position].reshape(-1)
        bin_index = (position_probs * num_bins).long().clamp_(0, num_bins - 1)
        count = torch.zeros(num_bins, dtype=torch.float64)
        pred_sum = torch.zeros(num_bins, dtype=torch.float64)
        target_sum = torch.zeros(num_bins, dtype=torch.float64)
        count.scatter_add_(0, bin_index, torch.ones_like(position_probs))
        pred_sum.scatter_add_(0, bin_index, position_probs)
        target_sum.scatter_add_(0, bin_index, position_targets)

        position_bins: list[dict[str, Optional[float] | int]] = []
        for bin_id in range(num_bins):
            bin_count = int(count[bin_id].item())
            if bin_count == 0:
                mean_predicted: Optional[float] = None
                mean_target: Optional[float] = None
            else:
                mean_predicted = float((pred_sum[bin_id] / bin_count).item())
                mean_target = float((target_sum[bin_id] / bin_count).item())
            position_bins.append(
                {
                    "bin": bin_id,
                    "lower": float(bin_id / num_bins),
                    "upper": float((bin_id + 1) / num_bins),
                    "count": bin_count,
                    "mean_predicted": mean_predicted,
                    "mean_target": mean_target,
                }
            )
        all_bins.append(position_bins)
    return all_bins


def evaluate_sts_calibration(
    *,
    logits: torch.Tensor,
    prefix_mask: torch.Tensor,
    temperatures: list[float],
    num_bins: int,
) -> dict:
    if logits.shape != prefix_mask.shape:
        raise ValueError(
            "STS eval logits / prefix_mask shape mismatch: "
            f"{tuple(logits.shape)} vs {tuple(prefix_mask.shape)}."
        )
    num_samples, gamma = logits.shape
    if num_samples == 0:
        raise ValueError("evaluate_sts_calibration requires at least one sample.")

    targets = prefix_mask.to(torch.float64)
    probs_before = survival_probabilities(logits=logits)
    probs_after = survival_probabilities(
        logits=logits,
        temperatures=temperatures,
    )
    brier_before = _brier_scores(probs=probs_before, targets=targets)
    brier_after = _brier_scores(probs=probs_after, targets=targets)
    bins_before = reliability_bins(
        probs=probs_before,
        targets=targets,
        num_bins=num_bins,
    )
    bins_after = reliability_bins(
        probs=probs_after,
        targets=targets,
        num_bins=num_bins,
    )

    per_position = []
    for position in range(gamma):
        position_targets = targets[:, position]
        per_position.append(
            {
                "position": position,
                "temperature": float(temperatures[position]),
                "num_samples": int(num_samples),
                "ece_before": expected_calibration_error(
                    probs=probs_before[:, position],
                    targets=position_targets,
                    num_bins=num_bins,
                ),
                "ece_after": expected_calibration_error(
                    probs=probs_after[:, position],
                    targets=position_targets,
                    num_bins=num_bins,
                ),
                "brier_before": float(brier_before[position]),
                "brier_after": float(brier_after[position]),
                "mean_predicted_survival_before": float(
                    probs_before[:, position].mean().item()
                ),
                "mean_predicted_survival_after": float(
                    probs_after[:, position].mean().item()
                ),
                "mean_target": float(position_targets.mean().item()),
                "reliability_bins_before": bins_before[position],
                "reliability_bins_after": bins_after[position],
            }
        )

    return {
        "num_samples": int(num_samples),
        "gamma": int(gamma),
        "num_bins": int(num_bins),
        "per_position": per_position,
    }


def fit_sts_temperatures(
    *,
    logits: torch.Tensor,
    prefix_mask: torch.Tensor,
    grid: torch.Tensor,
    num_bins: int = 15,
) -> dict[str, list[float]]:
    logits = logits.to(torch.float64)
    prefix_mask = prefix_mask.to(torch.float64)
    num_samples, gamma = logits.shape
    if num_samples == 0:
        raise ValueError("fit_sts_temperatures requires at least one sample.")
    grid_values = grid.to(torch.float64).tolist()

    temperatures: list[float] = []
    ece_before: list[float] = []
    ece_after: list[float] = []

    survival_at_one = torch.ones(num_samples, dtype=torch.float64)
    survival_fitted = torch.ones(num_samples, dtype=torch.float64)
    for position in range(gamma):
        position_logits = logits[:, position]
        position_target = prefix_mask[:, position]

        survival_at_one = survival_at_one * torch.sigmoid(position_logits)
        ece_before.append(
            expected_calibration_error(
                probs=survival_at_one,
                targets=position_target,
                num_bins=num_bins,
            )
        )

        best_temperature = grid_values[0]
        best_survival = survival_fitted * torch.sigmoid(
            position_logits / best_temperature
        )
        best_ece = expected_calibration_error(
            probs=best_survival, targets=position_target, num_bins=num_bins
        )
        for temperature in grid_values[1:]:
            candidate_survival = survival_fitted * torch.sigmoid(
                position_logits / temperature
            )
            candidate_ece = expected_calibration_error(
                probs=candidate_survival,
                targets=position_target,
                num_bins=num_bins,
            )
            if candidate_ece < best_ece:
                best_ece = candidate_ece
                best_temperature = temperature
                best_survival = candidate_survival

        temperatures.append(float(best_temperature))
        ece_after.append(float(best_ece))
        survival_fitted = best_survival

    return {
        "temperatures": temperatures,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }


def load_collected_shards(*, data_glob: str) -> tuple[torch.Tensor, torch.Tensor]:
    shard_paths = sorted(glob.glob(data_glob))
    if not shard_paths:
        raise ValueError(f"No STS data shards matched {data_glob!r}.")

    logits_shards: list[torch.Tensor] = []
    prefix_mask_shards: list[torch.Tensor] = []
    shard_gamma: Optional[int] = None
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        shard_logits = shard["logits"]
        shard_prefix_mask = shard["prefix_mask"]
        metadata = shard.get("metadata")
        if shard_logits.shape != shard_prefix_mask.shape:
            raise ValueError(
                f"Shard {shard_path!r} logits / prefix_mask shape mismatch: "
                f"{tuple(shard_logits.shape)} vs {tuple(shard_prefix_mask.shape)}."
            )
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Shard {shard_path!r} metadata must be a dict, got "
                    f"{type(metadata).__name__}."
                )
            metadata_gamma = metadata.get("gamma")
            if metadata_gamma is not None and int(metadata_gamma) != int(
                shard_logits.shape[1]
            ):
                raise ValueError(
                    f"Shard {shard_path!r} metadata gamma {int(metadata_gamma)} "
                    f"disagrees with tensor gamma {int(shard_logits.shape[1])}."
                )
            metadata_samples = metadata.get("num_samples")
            if metadata_samples is not None and int(metadata_samples) != int(
                shard_logits.shape[0]
            ):
                raise ValueError(
                    f"Shard {shard_path!r} metadata num_samples "
                    f"{int(metadata_samples)} disagrees with tensor samples "
                    f"{int(shard_logits.shape[0])}."
                )
        if shard_gamma is None:
            shard_gamma = int(shard_logits.shape[1])
        elif int(shard_logits.shape[1]) != shard_gamma:
            raise ValueError(
                f"Shard {shard_path!r} gamma {int(shard_logits.shape[1])} disagrees "
                f"with earlier shards' gamma {shard_gamma}."
            )
        logits_shards.append(shard_logits)
        prefix_mask_shards.append(shard_prefix_mask)

    return torch.cat(logits_shards, dim=0), torch.cat(prefix_mask_shards, dim=0)


def fit(
    *,
    data_glob: str,
    out: Path,
    num_bins: int = 15,
    gamma: Optional[int] = None,
    report_out: Optional[Path] = None,
    eval_data_glob: Optional[str] = None,
) -> None:
    logits, prefix_mask = load_collected_shards(data_glob=data_glob)
    resolved_gamma = int(logits.shape[1])
    if gamma is not None and gamma != resolved_gamma:
        raise ValueError(
            f"Collected shards have gamma={resolved_gamma} but --gamma={gamma}."
        )
    num_samples = int(logits.shape[0])

    result = fit_sts_temperatures(
        logits=logits,
        prefix_mask=prefix_mask,
        grid=default_temperature_grid(),
        num_bins=num_bins,
    )
    calibration = DSparkStsCalibration(
        temperatures=result["temperatures"],
        dataset=data_glob,
        num_samples=num_samples,
        ece_before=result["ece_before"],
        ece_after=result["ece_after"],
    )
    out.write_text(calibration.to_json(), encoding="utf-8")

    if report_out is not None:
        eval_logits = logits
        eval_prefix_mask = prefix_mask
        resolved_eval_data_glob = data_glob
        if eval_data_glob is not None:
            eval_logits, eval_prefix_mask = load_collected_shards(
                data_glob=eval_data_glob
            )
            resolved_eval_data_glob = eval_data_glob
            if int(eval_logits.shape[1]) != resolved_gamma:
                raise ValueError(
                    f"Eval shards have gamma={int(eval_logits.shape[1])} but "
                    f"fit shards have gamma={resolved_gamma}."
                )
        report = {
            "fit": {
                "data_glob": data_glob,
                "num_samples": num_samples,
                "gamma": resolved_gamma,
            },
            "eval": {
                "data_glob": resolved_eval_data_glob,
                **evaluate_sts_calibration(
                    logits=eval_logits,
                    prefix_mask=eval_prefix_mask,
                    temperatures=result["temperatures"],
                    num_bins=num_bins,
                ),
            },
        }
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        f"Fit STS temperatures over {num_samples} samples (gamma={resolved_gamma}) "
        f"-> {out}"
    )
    print("pos  temperature  ece_before  ece_after")
    for position in range(resolved_gamma):
        print(
            f"{position:>3}  {result['temperatures'][position]:>11.4f}  "
            f"{result['ece_before'][position]:>10.4f}  "
            f"{result['ece_after'][position]:>9.4f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Fit DSpark Sequential Temperature Scaling (STS) calibration "
        "temperatures from collected confidence shards."
    )
    parser.add_argument(
        "--data-glob",
        required=True,
        help="Glob of collected .pt shards, each a dict with [n, gamma] "
        "'logits' and 'prefix_mask' tensors.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output STS calibration JSON path.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of equal-width ECE bins.",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=None,
        help="Optional gamma override to validate the shards against.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional output JSON path for fit/eval calibration metrics.",
    )
    parser.add_argument(
        "--eval-data-glob",
        default=None,
        help="Optional held-out shard glob used for --report-out metrics. "
        "Defaults to --data-glob when omitted.",
    )
    args = parser.parse_args()
    if args.eval_data_glob is not None and args.report_out is None:
        parser.error("--eval-data-glob requires --report-out.")

    fit(
        data_glob=args.data_glob,
        out=args.out,
        num_bins=args.num_bins,
        gamma=args.gamma,
        report_out=args.report_out,
        eval_data_glob=args.eval_data_glob,
    )


if __name__ == "__main__":
    main()
