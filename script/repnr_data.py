
"""
Generate repetition-neuron datasets by sampling from a causal LM as Hiraoka's work (2024)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import regex as re
import torch
from transformers import GenerationConfig

from repnr_ind import utils


def count_overlap(text: str, query: str) -> int:
    return len(re.findall(query, text, overlapped=True))


def get_first_appearing_idx(text: str, query: str) -> int:
    return re.finditer(query, text).__next__().start(0)


def detect_repetition(line: Sequence[int], n: int, r: int, k: int) -> Tuple[List[int], int, int, int]:
    """Return the repeated n-gram and its positions if a repetition is detected."""
    sep = " "
    limit = len(line) - n + 1
    for i in range(0, limit):
        ngram = line[i : i + n]
        ngram_str = sep.join(map(str, ngram))
        window = line[max(0, i + n - r) : i + n]
        window_str = sep.join(map(str, window))
        if k > count_overlap(window_str, ngram_str):
            continue
        try:
            first = get_first_appearing_idx(window_str, ngram_str)
            first = len(window_str[:first].split())
            first += max(0, i + n - r)

            second_window = sep.join(map(str, line[first + 1 : i + n]))
            second = get_first_appearing_idx(second_window, ngram_str)
            second = len(second_window[:second].split())
            second += first + 1

            third_window = sep.join(map(str, line[second + 1 : i + n]))
            third = get_first_appearing_idx(third_window, ngram_str)
            third = len(third_window[:third].split())
            third += second + 1

            if (second - first) == (third - second):
                return list(ngram), first, second, third
        except StopIteration:
            continue
    return [], -1, -1, -1


def ensure_pad_token(model, tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


def generate_data_sample(
    model,
    tokenizer,
    n: int = 10,
    r: int = 100,
    k: int = 3,
    minimum_non_repetitive_affix: int = 50,
    num_random_sample_tokens: int = 10,
    num_greedy_generation_tokens: int = 200,
):
    generation_config_sample = GenerationConfig(
        max_new_tokens=num_random_sample_tokens,
        do_sample=True,
        temperature=1.0,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_config_greedy = GenerationConfig(
        max_new_tokens=num_greedy_generation_tokens,
        do_sample=False,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.inference_mode():
        inputs = tokenizer(" ", return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        sampled = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        extended = model.generate(sampled, generation_config=generation_config_greedy)

    tokens = extended[0].tolist()
    ngram, first, second, third = detect_repetition(tokens, n, r, k)
    if not ngram or second <= minimum_non_repetitive_affix:
        return None

    return {
        "promptIds": sampled[0].tolist(),
        "promptTokens": tokenizer.decode(sampled[0]),
        "firstPosition": first,
        "secondPosition": second,
        "thirdPosition": third,
        "ngramIds": ngram,
        "ngramTokens": tokenizer.decode(ngram),
        "50TokensBeforeRepeat": tokenizer.decode(tokens[max(0, second - 50) : second]),
        "50TokensAfterRepeat": tokenizer.decode(tokens[second : second + 50]),
        "generatedIds": tokens,
    }



def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "data" / "repetition_neuron_data"

    parser = argparse.ArgumentParser(description="Generate repetition-neuron data samples.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face access token (env HF_TOKEN).")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to attempt.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory for output JSONL files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--random-tokens", type=int, default=10, help="Random sampling token count.")
    parser.add_argument("--greedy-tokens", type=int, default=200, help="Greedy generation token count.")
    parser.add_argument("--n", type=int, default=10, help="Repeated n-gram length.")
    parser.add_argument("--r", type=int, default=100, help="Search range for repetitions.")
    parser.add_argument("--k", type=int, default=3, help="Minimum repetition count.")
    parser.add_argument("--min-affix", type=int, default=50, help="Minimum prefix length before repetition.")
    return parser.parse_args()


def sanitize_model_name(model_name: str) -> str:
    tail = model_name.split("/")[-1]
    return tail.replace(" ", "_")


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("HF token is required (pass --hf-token or set HF_TOKEN env).")

    utils.seed_everything(args.seed)
    model, tokenizer = utils.load_model(args.model_name, token=args.hf_token, seed=args.seed)
    ensure_pad_token(model, tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_model_name(args.model_name)}.jsonl"
    output_path = args.output_dir / filename

    from tqdm import tqdm

    written = 0
    for _ in tqdm(range(args.num_samples), desc="Collect samples"):
        sample = generate_data_sample(
            model,
            tokenizer,
            n=args.n,
            r=args.r,
            k=args.k,
            minimum_non_repetitive_affix=args.min_affix,
            num_random_sample_tokens=args.random_tokens,
            num_greedy_generation_tokens=args.greedy_tokens,
        )
        if not sample:
            continue
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")
        written += 1

    print(f"Wrote {written} samples to {output_path}")


if __name__ == "__main__":
    main()
