from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple


@dataclass
class ModelConfig:
    """HF model metadata used across every experiment."""
    name: str = "meta-llama/Llama-3.1-8B"
    token: str | None = None
    seed: int = 42


@dataclass
class DatasetConfig:
    """
    Configuration for both repetition-ranking data and ICL evaluation data.
    `ranking_path` points to the JSONL used to discover repetition neurons.
    `icl_root`/`icl_tasks`/`icl_shot` specify the in-context learning datasets.
    """
    ranking_path: Path
    base_samples: int = 1000
    eval_samples: int = 100
    max_range: int = 30
    repetition_position: str = "second"  # first/second/third in the JSON schema
    icl_root: Path | None = None
    icl_tasks: Sequence[str] = ()
    icl_shot: int = 10
    icl_task_type: str = "abstract"


@dataclass
class AblationConfig:
    """Hyper-parameters shared by neuron/head/joint ablations."""
    neurons_to_ablate: Sequence[int] = (50, 100, 150, 200, 250)
    percent_heads: Sequence[int] = (1, 3, 5, 7, 10)
    segment_ranges: Sequence[Tuple[float, float]] = (
        (0.0, 0.2),
        (0.4, 0.6),
        (0.8, 1.0),
    )
    ngram: int = 10
    search_range: int = 100
    repeats: int = 3
    prefix_length: int = 50
    random_tokens: int = 10
    greedy_tokens: int = 200


@dataclass
class ExperimentConfig:
    """Full experiment description tying every knob together."""
    model: ModelConfig
    dataset: DatasetConfig
    ablation: AblationConfig
    output_path: Path
    notes: str = ""
