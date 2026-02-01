from pathlib import Path
import json
import random

from repnr_ind.experiments.config import (
    AblationConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
)
from repnr_ind.experiments.pipeline import load_icl_datasets, prepare_model
from repnr_ind import induction_heads, utils


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
            neurons_to_ablate=(100, 250),  # unused in head-only run
            percent_heads=(1, 3),
        ),
        output_path=Path("data/ablation_head_only.jsonl"),
    )

    icl_datasets_full = load_icl_datasets(cfg)
    SAMPLE_LIMIT = 10  # for testing only, remove to run full dataset
    icl_datasets = {
    task: random.sample(records, k=min(SAMPLE_LIMIT, len(records)))
    for task, records in icl_datasets_full.items()
}

    model, tokenizer = prepare_model(cfg)
    head_scores = induction_heads.compute_prefix_matching_scores(model, tokenizer)

    hd_result = {}
    for task, dataset in icl_datasets.items():
        nor_answers = _baseline_answers(
            model, tokenizer, dataset, cfg.model.seed, cfg.dataset.icl_task_type
        )
        ab_hd_result = induction_heads.ablate_attention_heads_8(
            model=model,
            tokenizer=tokenizer,
            avg_scores=head_scores,
            test_dataset=dataset,
            percent_list=list(cfg.ablation.percent_heads),
            seed=cfg.model.seed,
        )
        hd = utils.extract_answers_ab(ab_hd_result, task=cfg.dataset.icl_task_type)
        hd_result[task] = {
            "ground_truths": [item["ground_truth"] for item in dataset],
            "nor_": nor_answers,
            "hd": hd,
        }

    accs = induction_heads.compute_head_ablation_accuracy(hd_result)
    for task, data in accs.items():
        print(f"[{task}]")
        print(f"  nor_acc: {data['nor_acc']:.3f}")
        for mode in ("induction", "random"):
            for p in cfg.ablation.percent_heads:
                val = data["hd_acc"][mode].get(p)
                if val is not None:
                    print(f"  {mode} {p}%: {val:.3f}")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w", encoding="utf-8") as f:
        for task, res in hd_result.items():
            f.write(json.dumps({task: res}) + "\n")


if __name__ == "__main__":
    main()
