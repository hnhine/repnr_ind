from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from .. import utils


def summarize_neuron_ablation(records: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.get("experiment") != "neuron_segment_ablation":
            continue
        rows.append(
            {
                "mode": record["mode"],
                "neurons": record["neurons"],
                "segment": f"{record['segment']['start']}->{record['segment']['end']}",
                "num_repetitions": record["num_repetitions"],
            }
        )
    return pd.DataFrame(rows)


def summarize_head_ablation(records: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.get("experiment") != "head_ablation":
            continue
        rows.append(
            {
                "mode": record["mode"],
                "percent": record["percent"],
                "num_repetitions": record["num_repetitions"],
            }
        )
    return pd.DataFrame(rows)


def plot_prefix_matching_scores(head_scores: Sequence[torch.Tensor], save_path: Path | None = None) -> None:
    matrix = torch.stack([layer.cpu() for layer in head_scores]).numpy()
    plt.figure(figsize=(8, 4))
    sns.heatmap(matrix, cmap="viridis", cbar=True)
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Prefix-matching scores")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def build_class_dataframe(ab_dataset: Sequence[dict], answers: Sequence[str], ablation_dict: dict, prefix: str) -> pd.DataFrame:
    ground_truths = [item["ground_truth"] for item in ab_dataset]
    queries = [item["query"] for item in ab_dataset]
    df = pd.DataFrame({"query": queries, "ground_truth": ground_truths, "baseline": answers})
    for mode, mode_results in ablation_dict.items():
        for key, outputs in mode_results.items():
            column = f"{prefix}_{mode}_{key}"
            df[column] = outputs
    return df


def evaluate_accuracy_per_class(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    overview = utils.analyze_class_overview(df.to_dict(orient="list"))
    return overview
