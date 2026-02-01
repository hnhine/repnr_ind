import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import GenerationConfig

from repnr_ind import induction_heads, repetition_neurons, utils


def parse_segment_ranges(spec: str) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            start_str, end_str = chunk.split("-")
            start_val = float(start_str)
            end_val = float(end_str)
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(f"Malformed segment range '{chunk}'") from exc
        if not (0.0 <= start_val < end_val <= 1.0):
            raise argparse.ArgumentTypeError(f"Segment '{chunk}' must satisfy 0.0 <= start < end <= 1.0")
        ranges.append((start_val, end_val))
    if not ranges:
        raise argparse.ArgumentTypeError("At least one segment range is required.")
    return ranges


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_dir = repo_root / "data" / "repetition_neuron_data"
    default_output = repo_root / "data" / "text_generation_results.jsonl"

    parser = argparse.ArgumentParser(description="Run text-generation ablation experiments.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"))
    parser.add_argument("--data-path", type=Path, help="Repetition dataset JSONL. Defaults to data/repetition_neuron_data/<model>.jsonl")
    parser.add_argument("--base-samples", type=int, default=1000, help="Samples used to rank repetition neurons.")
    parser.add_argument("--eval-samples", type=int, default=100, help="Samples evaluated during ablations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment-ranges", type=parse_segment_ranges, default=parse_segment_ranges("0-0.2,0.4-0.6,0.8-1.0"))
    parser.add_argument("--neurons-to-ablate", type=int, nargs="+", default=[50, 100, 150, 200, 250])
    parser.add_argument("--percent-heads", type=int, nargs="+", default=[1, 3, 5, 7, 10])
    parser.add_argument("--ngram", type=int, default=10)
    parser.add_argument("--search-range", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefix-length", type=int, default=50, help="Tokens kept before greedy continuation.")
    parser.add_argument("--random-tokens", type=int, default=10, help="Number of random tokens in dataset creation.")
    parser.add_argument("--greedy-tokens", type=int, default=200, help="Number of greedy tokens in dataset creation.")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--no-neuron-ablation", action="store_true", help="Skip repetition neuron ablations.")
    parser.add_argument("--no-head-ablation", action="store_true", help="Skip induction-head ablations.")
    return parser.parse_args()


def ensure_pad_token(model, tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


def dataset_slices(dataset: Sequence[dict], base_samples: int, eval_samples: int) -> Tuple[List[dict], List[dict]]:
    if len(dataset) < base_samples + eval_samples:
        raise ValueError(
            f"Dataset has {len(dataset)} items but requires at least {base_samples + eval_samples} "
            "for base + evaluation splits."
        )
    base = dataset[:base_samples]
    evaluation = dataset[base_samples : base_samples + eval_samples]
    return base, evaluation


def select_segment_neurons(
    sorted_neurons: Sequence[dict],
    num_layers: int,
    start_frac: float,
    end_frac: float,
) -> List[Tuple[int, int]]:
    start_layer = max(0, min(num_layers - 1, int(num_layers * start_frac)))
    end_layer = max(start_layer + 1, min(num_layers, int(num_layers * end_frac)))
    selected: List[Tuple[int, int]] = []
    for entry in sorted_neurons:
        layer_idx, neuron_idx = entry["neuron"]
        if start_layer <= layer_idx < end_layer:
            selected.append((layer_idx, neuron_idx))
    return selected


def evaluate_dataset(
    model,
    tokenizer,
    dataset: Sequence[dict],
    detect_kwargs: dict,
    new_tokens: int,
    progress_desc: str,
) -> Tuple[List[dict], int]:
    results: List[dict] = []
    repetitions = 0
    generation_config = GenerationConfig(
        max_new_tokens=new_tokens,
        do_sample=False,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )
    with torch.inference_mode():
        iterator = tqdm(
            dataset,
            desc=progress_desc,
            unit="sample",
            leave=False,
        )
        for idx, sample in enumerate(iterator):
            ids = sample.get("generatedIds")
            if not ids:
                continue
            _, _, second_pos, _ = repetition_neurons.detect_repetition(ids, **detect_kwargs)
            if second_pos <= 0:
                continue
            prompt_ids = torch.LongTensor([ids[:second_pos]]).to(model.device)
            outputs = model.generate(prompt_ids, generation_config=generation_config)
            gens = outputs[0].tolist()
            ngram, *_ = repetition_neurons.detect_repetition(gens, **detect_kwargs)
            label = "REPL" if ngram else "NORE"
            if ngram:
                repetitions += 1
            text = tokenizer.decode(gens)
            results.append({"sample_index": idx, "label": label, "text": text})
    return results, repetitions


def run_neuron_ablation(
    model,
    tokenizer,
    dataset: Sequence[dict],
    sorted_neurons: Sequence[dict],
    segment_ranges: Sequence[Tuple[float, float]],
    neurons_to_ablate: Sequence[int],
    detect_kwargs: dict,
    new_tokens: int,
) -> List[dict]:
    num_layers = len(model.model.layers)
    experiments: List[dict] = []
    for mode in ("top", "random"):
        for count in neurons_to_ablate:
            segment_iterator = tqdm(
                segment_ranges,
                desc=f"Neuron ablation ({mode}, {count} neurons)",
                unit="segment",
            )
            for start, end in segment_iterator:
                segment_neurons = select_segment_neurons(sorted_neurons, num_layers, start, end)
                if not segment_neurons:
                    continue
                if mode == "top":
                    chosen = segment_neurons[:count]
                else:
                    rng = torch.Generator().manual_seed(0)
                    idxs = torch.randperm(len(segment_neurons), generator=rng).tolist()
                    chosen = [segment_neurons[i] for i in idxs[:count]]
                neurons_by_layer: Dict[int, List[int]] = {}
                for layer_idx, neuron_idx in chosen:
                    neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)

                hooks = [
                    repetition_neurons.Deactivator(
                        model.model.layers[layer_idx].mlp.act_fn, neuron_ids, mode="all"
                    )
                    for layer_idx, neuron_ids in neurons_by_layer.items()
                ]
                try:
                    samples, num_rep = evaluate_dataset(
                        model,
                        tokenizer,
                        dataset,
                        detect_kwargs,
                        new_tokens,
                        progress_desc=f"Eval neurons {mode} {count} ({start:.1f}-{end:.1f})",
                    )
                finally:
                    for hook in hooks:
                        hook.release()
                experiments.append(
                    {
                        "experiment": "neuron_segment_ablation",
                        "mode": mode,
                        "neurons": count,
                        "segment": {"start": start, "end": end},
                        "num_repetitions": num_rep,
                        "samples": samples,
                    }
                )
    return experiments


def run_head_ablation(
    model,
    tokenizer,
    dataset: Sequence[dict],
    detect_kwargs: dict,
    avg_scores: Sequence[torch.Tensor],
    percent_list: Sequence[int],
    new_tokens: int,
    seed: int,
) -> List[dict]:
    experiments: List[dict] = []
    num_layers = len(avg_scores)
    num_heads = avg_scores[0].shape[0]
    masks = induction_heads.build_head_masks(avg_scores, num_layers, num_heads, percent_list, random_seed=seed)

    for mode in ("induction", "random"):
        percent_iterator = tqdm(
            percent_list,
            desc=f"Head ablation ({mode})",
            unit="percent",
        )
        for percent in percent_iterator:
            mask = masks[mode][percent]
            block_config = {
                layer: [head for head, keep in enumerate(mask[layer].tolist()) if keep == 0.0]
                for layer in range(num_layers)
            }
            orig_blocks = induction_heads.disable_Wo_heads(model, block_config)
            try:
                samples, num_rep = evaluate_dataset(
                    model,
                    tokenizer,
                    dataset,
                    detect_kwargs,
                    new_tokens,
                    progress_desc=f"Eval heads {mode} {percent}%",
                )
            finally:
                induction_heads.restore_Wo_heads(model, orig_blocks)
            experiments.append(
                {
                    "experiment": "head_ablation",
                    "mode": mode,
                    "percent": percent,
                    "num_repetitions": num_rep,
                    "samples": samples,
                }
            )
    return experiments


def main() -> None:
    args = parse_args()
    utils.seed_everything(args.seed)
    print(f"Using global seed {args.seed}")

    data_path = args.data_path
    if data_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data" / "repetition_neuron_data"
        tail = args.model_name.split("/")[-1].replace(" ", "_")
        data_path = data_dir / f"{tail}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find dataset at {data_path}")

    dataset = utils.load_data(str(data_path))
    base_dataset, eval_dataset = dataset_slices(dataset, args.base_samples, args.eval_samples)

    model, tokenizer = utils.load_model(args.model_name, token=args.hf_token, seed=args.seed)
    ensure_pad_token(model, tokenizer)

    detect_kwargs = {"n": args.ngram, "r": args.search_range, "k": args.repeats}
    new_tokens = max(1, args.random_tokens + args.greedy_tokens - args.prefix_length)

    records: List[dict] = []
    if not args.no_neuron_ablation:
        sorted_neurons = repetition_neurons.find_sorted_neurons(
            model, tokenizer, base_data=base_dataset, seed=args.seed
        )
        neuron_results = run_neuron_ablation(
            model,
            tokenizer,
            eval_dataset,
            sorted_neurons,
            args.segment_ranges,
            args.neurons_to_ablate,
            detect_kwargs,
            new_tokens,
        )
        records.extend(neuron_results)

    if not args.no_head_ablation:
        avg_scores = induction_heads.compute_prefix_matching_scores(model, tokenizer)
        head_results = run_head_ablation(
            model,
            tokenizer,
            eval_dataset,
            detect_kwargs,
            avg_scores,
            args.percent_heads,
            new_tokens,
            args.seed,
        )
        records.extend(head_results)

    if not records:
        raise SystemExit("No experiments were executed (all ablations skipped?).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} experiment summaries to {args.output}")


if __name__ == "__main__":
    main()
