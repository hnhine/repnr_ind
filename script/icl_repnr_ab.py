from pathlib import Path
import random

from repnr_ind.experiments.config import (
    AblationConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
)
from repnr_ind.experiments.pipeline import (
    build_neuron_layer_map,
    compute_sorted_neurons,
    load_icl_datasets,
    prepare_datasets,
    prepare_model,
    run_neuron_ablation_icl,
)
from repnr_ind import utils


def main() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(name="meta-llama/Llama-3.1-8B", token=None, seed=42),
        dataset=DatasetConfig(
            ranking_path=Path("data/repetition_neuron_data/Llama-3.1-8B.jsonl"),
            base_samples=100,
            #eval_samples=3,
            icl_root=Path("data/abstract_icl"),
            icl_tasks=("rep", "rec", "ceb", "wsq1", "wsq2"),
            icl_shot=10,
            icl_task_type="abstract",
        ),
        ablation=AblationConfig(
            neurons_to_ablate=(100,250),
            percent_heads=(1,3),  # unused in neuron-only run
        ),
        output_path=Path("data/ablation_neuron_only.jsonl"),
    )

    ranking_data, _ = prepare_datasets(cfg)
    icl_datasets_full = load_icl_datasets(cfg)
    SAMPLE_LIMIT = 10 #for testing only, remove to run full dataset
    icl_datasets = {
    task: random.sample(records, k=min(SAMPLE_LIMIT, len(records)))
    for task, records in icl_datasets_full.items()
}
    model, tokenizer = prepare_model(cfg)
    sorted_neurons = compute_sorted_neurons(model, tokenizer, ranking_data, cfg)
    _ = build_neuron_layer_map(sorted_neurons, len(model.model.layers))  # optional

    neuron_records = run_neuron_ablation_icl(
        model=model,
        tokenizer=tokenizer,
        dataset_map=icl_datasets,
        sorted_neurons=sorted_neurons,
        cfg=cfg,
    )

    for rec in neuron_records:
        print(f"{rec['task']} -> modes: {list(rec['nr_layer'].keys())}")

    # compute Foo/Bar recall
    results_dict = {rec["task"]: rec for rec in neuron_records}
    last_K = cfg.ablation.neurons_to_ablate[-1]
    metrics = utils.compute_ablation_accuracies(results_dict, mode="top", K=last_K) #report the result of ablating repetition neurons
    for task, segments in metrics.items():
        print(f"[{task}]")
        for segment, values in segments.items():
            if segment in ("Foo_nor", "Bar_nor"):
                print(f"  {segment}: {values:.3f}")
            else:
                print(f"  {segment}: Foo={values['Foo']:.3f}, Bar={values['Bar']:.3f}")

    # saving example (optional)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with cfg.output_path.open("w", encoding="utf-8") as f:
        for rec in neuron_records:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
