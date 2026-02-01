from pathlib import Path
import json
import random

from repnr_ind.experiments.config import (
    AblationConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
)
from repnr_ind.experiments.pipeline import (
    compute_head_scores,
    compute_sorted_neurons,
    load_icl_datasets,
    prepare_datasets,
    prepare_model,
)
from repnr_ind import utils


def _baseline_answers(model, tokenizer, dataset, seed, task_type: str):
    answers, _ = utils.generate_answer(model, tokenizer, dataset, seed, task=task_type)
    if task_type == "wmt":
        return utils.extract_answers_nor_wmt(answers)
    return utils.extract_answers_nor(answers)


def main() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(name="meta-llama/Llama-3.1-8B", token=None, seed=42),
        dataset=DatasetConfig(
            ranking_path=Path("data/repetition_neuron_data/Llama-3.1-8B.jsonl"),
            base_samples=100,
            eval_samples=3,
            icl_root=Path("data/abstract_icl"),
            icl_tasks=("rep", "rec", "ceb", "wsq1", "wsq2"),
            icl_shot=10,
            icl_task_type="abstract",
        ),
        ablation=AblationConfig(
            neurons_to_ablate=(100, 250),
            percent_heads=(1, 3),
        ),
        output_path=Path("data/ablation_joint_2phase.jsonl"),
    )

    ranking_data, _ = prepare_datasets(cfg)
    icl_datasets_full = load_icl_datasets(cfg)
    SAMPLE_LIMIT = 10  # for testing only, remove to run full dataset
    icl_datasets = {
        task: random.sample(records, k=min(SAMPLE_LIMIT, len(records)))
        for task, records in icl_datasets_full.items()
    }

    model, tokenizer = prepare_model(cfg)
    sorted_neurons = compute_sorted_neurons(model, tokenizer, ranking_data, cfg)
    head_scores = compute_head_scores(model, tokenizer)

    joint_records = []
    joint_results = {}
    for task, dataset in icl_datasets.items():
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
            progress="outer",
        )
        nor_answers = _baseline_answers(
            model, tokenizer, dataset, cfg.model.seed, cfg.dataset.icl_task_type
        )
        joint_results[task] = {
            "ground_truths": [item["ground_truth"] for item in dataset],
            "nor_": nor_answers,
            "ab_dual": results,
        }
        joint_records.append(
            {
                "experiment": "joint_ablation",
                "task": task,
                "ground_truths": [item["ground_truth"] for item in dataset],
                "nor_": nor_answers,
                "ab_dual": results,
            }
        )

    dual_accs = utils.compute_dual_ablation_accuracies_by_class(joint_results)
    for task, info in dual_accs.items():
        print(f"[{task}]")
        baseline = ", ".join(f"{k}={v:.3f}" for k, v in info["baseline"].items())
        print(f"  baseline: {baseline}")
        for rep_mode, rep_block in info["ablation"].items():
            for mask_mode, mask_block in rep_block.items():
                for k, pct_block in mask_block.items():
                    for pct, seg_map in pct_block.items():
                        segs = ", ".join(
                            f"{seg}: " + ", ".join(f"{cls}={acc:.3f}" for cls, acc in cls_map.items())
                            for seg, cls_map in seg_map.items()
                        )
                        print(f"  {rep_mode}/{mask_mode}/k={k}/p={pct}% -> {segs}")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w", encoding="utf-8") as f:
        for rec in joint_records:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
