from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm
from .. import induction_heads, repetition_neurons, utils
from .config import ExperimentConfig
from .data import read_jsonl, split_dataset, write_jsonl


def ensure_pad_token(model, tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


def prepare_model(cfg: ExperimentConfig):
    utils.seed_everything(cfg.model.seed)
    model, tokenizer = utils.load_model(cfg.model.name, token=cfg.model.token, seed=cfg.model.seed)
    ensure_pad_token(model, tokenizer)
    return model, tokenizer


def prepare_datasets(cfg: ExperimentConfig) -> Tuple[List[dict], List[dict]]:
    records = read_jsonl(cfg.dataset.ranking_path)
    return split_dataset(records, cfg.dataset.base_samples, cfg.dataset.eval_samples)


def load_icl_datasets(cfg: ExperimentConfig) -> Dict[str, List[dict]]:
    if not cfg.dataset.icl_tasks:
        return {}
    root = cfg.dataset.icl_root or Path("data/abstract_icl")
    datasets: Dict[str, List[dict]] = {}
    for task in cfg.dataset.icl_tasks:
        path = root / f"{task}_{cfg.dataset.icl_shot}shot.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"ICL dataset not found: {path}")
        datasets[task] = utils.load_data(str(path))
    return datasets


def compute_sorted_neurons(model, tokenizer, base_dataset, cfg: ExperimentConfig):
    return repetition_neurons.find_sorted_neurons(
        model,
        tokenizer,
        base_data=base_dataset,
        max_range=cfg.dataset.max_range,
        position=cfg.dataset.repetition_position,
        seed=cfg.model.seed,
    )


def compute_head_scores(model, tokenizer):
    return induction_heads.compute_prefix_matching_scores(model, tokenizer)


def build_neuron_layer_map(sorted_neurons, num_layers):
    mapping: Dict[float, List[dict]] = {}
    for neuron_info in sorted_neurons:
        layer_idx, _ = neuron_info["neuron"]
        rel = layer_idx / float(num_layers)
        mapping.setdefault(rel, []).append(neuron_info)
    return mapping


def run_neuron_ablation_icl(
    model,
    tokenizer,
    dataset_map: Dict[str, List[dict]],
    sorted_neurons: Sequence[dict],
    cfg: ExperimentConfig,
) -> List[dict]:
    if not dataset_map:
        return []
    task_label = "Neuron-only tasks"
    records: List[dict] = []
    for task, dataset in tqdm(list(dataset_map.items()), desc=task_label):
        ground_truths = [item["ground_truth"] for item in dataset]
        nor_answers, _ = utils.generate_answer(
            model,
            tokenizer,
            dataset,
            cfg.model.seed,
            task=cfg.dataset.icl_task_type,
        )
        if cfg.dataset.icl_task_type == "wmt":
            nor = utils.extract_answers_nor_wmt(nor_answers)
        else:
            nor = utils.extract_answers_nor(nor_answers)

        ab_results = repetition_neurons.run_segment_ablation_study_2phase(
            model,
            tokenizer,
            dataset,
            sorted_neurons,
            segment_ranges=list(cfg.ablation.segment_ranges),
            neurons_to_ablate_list=list(cfg.ablation.neurons_to_ablate),
            seed=cfg.model.seed,
            task=cfg.dataset.icl_task_type,
        )
        nr_layer = utils.extract_answers_ab_layer(ab_results, task=cfg.dataset.icl_task_type)
        records.append(
            {
                "experiment": "neuron_ablation",
                "task": task,
                "ground_truths": ground_truths,
                "nr_layer": nr_layer,
                "nor_": nor,
            }
        )
    return records


def run_head_ablation_icl(
    model,
    tokenizer,
    dataset_map: Dict[str, List[dict]],
    head_scores: Sequence[torch.Tensor],
    sorted_neurons: Sequence[dict],
    cfg: ExperimentConfig,
    neuron_mapping: Dict[float, List[dict]] | None = None,
) -> List[dict]:
    if not dataset_map:
        return []
    mapping = neuron_mapping or build_neuron_layer_map(sorted_neurons, len(model.model.layers))
    num_layers = len(head_scores)
    num_heads = head_scores[0].shape[0]
    masks = induction_heads.build_head_masks(
        head_scores, num_layers, num_heads, cfg.ablation.percent_heads, cfg.model.seed
    )
    results: List[dict] = []
    for task, dataset in tqdm(list(dataset_map.items()), desc="Head-only tasks"):
        for mode in ("induction", "random"):
            for percent in cfg.ablation.percent_heads:
                head_mask = masks[mode][percent]
                segments = utils.conduct_segment_dual_ablation_2phase(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    neurons_by_layer_position=mapping,
                    segment_ranges=list(cfg.ablation.segment_ranges),
                    neurons_to_ablate=0,
                    modeNr="top",
                    head_mask=head_mask,
                    shot=cfg.dataset.icl_shot,
                    seed=cfg.model.seed,
                    task=cfg.dataset.icl_task_type,
                )
                results.append(
                    {
                        "experiment": "head_ablation",
                        "task": task,
                        "mode": mode,
                        "percent": percent,
                        "segments": segments,
                    }
                )
    return results


def run_joint_ablation(
    model,
    tokenizer,
    dataset_map: Dict[str, List[dict]],
    sorted_neurons: Sequence[dict],
    head_scores: Sequence[torch.Tensor],
    cfg: ExperimentConfig,
) -> List[dict]:
    if not dataset_map:
        return []
    joint_runs: List[dict] = []
    for task, dataset in tqdm(list(dataset_map.items()), desc="Joint ablation tasks"):
        results = utils.run_dual_ablation_study(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            sortedNeurons=sorted_neurons,
            id_avg_scores=head_scores,
            percent_list=list(cfg.ablation.percent_heads),
            segment_ranges=list(cfg.ablation.segment_ranges),
            neurons_to_ablate_list=list(cfg.ablation.neurons_to_ablate),
            shot=cfg.dataset.icl_shot,
            seed=cfg.model.seed,
            task=cfg.dataset.icl_task_type,
            dataset_name=task,
        )
        joint_runs.append(
            {
                "experiment": "joint_ablation",
                "task": task,
                "results": results,
            }
        )
    return joint_runs


def run_pipeline(cfg: ExperimentConfig) -> Path:
    base_dataset, _ = prepare_datasets(cfg)
    icl_datasets = load_icl_datasets(cfg)
    model, tokenizer = prepare_model(cfg)
    sorted_neurons = compute_sorted_neurons(model, tokenizer, base_dataset, cfg)
    head_scores = compute_head_scores(model, tokenizer)
    neuron_mapping = build_neuron_layer_map(sorted_neurons, len(model.model.layers))

    neuron_records = run_neuron_ablation_icl(
        model,
        tokenizer,
        icl_datasets,
        sorted_neurons,
        cfg,
        neuron_mapping=neuron_mapping,
    )
    head_records = run_head_ablation_icl(
        model,
        tokenizer,
        icl_datasets,
        head_scores,
        sorted_neurons,
        cfg,
        neuron_mapping=neuron_mapping,
    )
    joint_records = run_joint_ablation(model, tokenizer, icl_datasets, sorted_neurons, head_scores, cfg)

    payload: List[dict] = [
        {
            "type": "metadata",
            "model": cfg.model.name,
            "ranking_dataset": str(cfg.dataset.ranking_path),
            "icl_tasks": list(cfg.dataset.icl_tasks),
            "notes": cfg.notes,
        },
        *neuron_records,
        *head_records,
        *joint_records,
    ]
    write_jsonl(cfg.output_path, payload)
    return cfg.output_path
