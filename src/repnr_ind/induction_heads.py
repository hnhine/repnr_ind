import numpy as np
import torch
import json
import functools
from tqdm import tqdm
import math
import random
import tempfile
import shutil
import os
import torch.nn.functional as F
from types import MethodType
import matplotlib.pyplot as plt
from . import utils


def compute_prefix_matching_scores(model, tokenizer, num_sequences=5, sequence_length=50, device=0):
    """
    Computes prefix matching scores for all attention heads in a transformer model
    using synthetic repeated token sequences.
    
    Returns:
        avg_scores (List[Tensor]): One tensor per layer of shape (num_heads,)
    """
    try:
        merges = tokenizer.backend_tokenizer.model.merges
    except AttributeError:
        # fallback to tokenizer.json
        tmp_dir = tempfile.mkdtemp()
        tokenizer.save_pretrained(tmp_dir)
        with open(os.path.join(tmp_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        merges = tokenizer_data["model"]["merges"]
        shutil.rmtree(tmp_dir)

    # --- Step 2: Build ranked BPE vocab ---
    seen = set()
    ranked_list = []
    for merge_string in merges:
        bpe_token = ''.join(merge_string).replace(' ', '')
        if bpe_token not in seen:
            seen.add(bpe_token)
            ranked_list.append(bpe_token)

    ranked_dict = {
        rank: tokenizer.convert_tokens_to_ids(token) for rank, token in enumerate(ranked_list)
    }

    # --- Step 3: Token filtering and model metadata ---
    vocab_size = len(ranked_list)
    exclude = int(0.04 * vocab_size)
    rank_choice_list = np.arange(exclude, vocab_size - exclude)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    all_scores = []

    with torch.no_grad():
        for seed in tqdm(range(num_sequences), desc="Generating sequences"):
            torch.manual_seed(seed)
            selected_ranks = np.random.choice(rank_choice_list, size=sequence_length, replace=False)
            token_ids = [tokenizer.bos_token_id] + [ranked_dict[rank] for rank in selected_ranks]
            input_ids = torch.tensor([token_ids], device=device)

            repeated_body = input_ids[:, 1:].repeat(3, 1).view(1, -1)
            repeated_ids = torch.cat([input_ids, repeated_body], dim=-1)
            assert repeated_ids.shape[1] == 4 * sequence_length + 1

            outputs = model(input_ids=repeated_ids, output_attentions=True)
            attentions = outputs.attentions

            attn_matrix = torch.zeros((num_layers, num_heads), device=device)
            for layer in range(num_layers):
                attn = attentions[layer].squeeze(0)  # (head, tgt, src)

                score = torch.zeros(num_heads, device=device)
                count = 0
                for i in range(sequence_length + 1, 4 * sequence_length + 1):
                    token_pos = i % sequence_length
                    repeat_steps = i // sequence_length
                    prev_positions = [(repeat_num * sequence_length + token_pos + 1) for repeat_num in range(repeat_steps)]
                    score += attn[:, i, prev_positions].sum(dim=1)
                    count += len(prev_positions)

                attn_matrix[layer] = score / count

            all_scores.append(attn_matrix.unsqueeze(0))

    final = torch.cat(all_scores, dim=0)
    mean = final.mean(dim=0)

    avg_scores = [mean[layer_idx].clone().detach().cpu() for layer_idx in range(num_layers)]
    return avg_scores

def build_head_masks(avg_scores, num_layers, num_heads, percent_list=[1, 3, 5, 7, 10], random_seed=42):
    """
    Build induction and random ablation masks exactly as in Sec. 4.3:

    - Compute n_ablate = ceil(num_layers * num_heads * (p/100))
    - Select the top-n_ablate heads by prefix-matching score using a single top-k over
      the flattened [layers×heads] tensor.
    - For random baseline, ablate the same count per-layer, but choose heads uniformly at random.

    Returns:
        masks: {
          "induction": {p: Tensor[num_layers, num_heads]},
          "random":    {p: Tensor[num_layers, num_heads]}
        }
    """
    masks = {"induction": {}, "random": {}}

    # Flatten all layer-head scores into a single vector
    all_flat = torch.cat([s for s in avg_scores]) 

    for p in percent_list:
        n_ablate = math.ceil(num_layers * num_heads * (p / 100))
        _, topk_idx = torch.topk(all_flat, n_ablate)

        # Initialize induction mask: 1 = keep, 0 = ablate
        induction_mask = torch.ones(num_layers, num_heads)
        for idx in topk_idx.tolist():
            layer = idx // num_heads
            head  = idx % num_heads
            induction_mask[layer, head] = 0.0
        masks["induction"][p] = induction_mask

        rnd_mask = torch.ones_like(induction_mask)
        rng = random.Random(random_seed)
        for layer in range(num_layers):
            k = int((induction_mask[layer] == 0.0).sum().item())
            if k > 0:
                available_heads = list(range(num_heads))
                selected = rng.sample(available_heads, k)
                for head in selected:
                    rnd_mask[layer, head] = 0.0
        masks["random"][p] = rnd_mask

    return masks
def disable_Wo_heads(model, block_config):
    """
    Zero out each block Wo_h in o_proj.weight.
    Return {(layer_idx, head_idx): original_block_tensor} to restore.
    """
    original = {}
    for layer_idx, heads in block_config.items():
        layer = model.model.layers[layer_idx].self_attn
        Dh    = layer.head_dim
        W     = layer.o_proj.weight    # nn.Parameter [d_out, H*Dh]
        for h in heads:
            s, e = h*Dh, (h+1)*Dh
            original[(layer_idx,h)] = W[:, s:e].data.clone()
            W[:, s:e].data.zero_()
    return original

def restore_Wo_heads(model, original):
    """
    Use `original` to restore Wo_h.
    """
    for (layer_idx,h), block in original.items():
        layer = model.model.layers[layer_idx].self_attn
        Dh    = layer.head_dim
        s, e  = h*Dh, (h+1)*Dh
        layer.o_proj.weight.data[:, s:e] = block
def ablate_attention_heads_8(model, tokenizer, avg_scores, test_dataset, percent_list=None, seed=42):
    print("Ablation induction heads ...")

    if percent_list is None:
        percent_list = [1, 3, 5, 7, 10]

    utils.seed_everything(seed)

    num_layers = len(avg_scores)
    num_heads = avg_scores[0].shape[0]
    masks = build_head_masks(avg_scores, num_layers, num_heads, percent_list, random_seed=seed)

    results = {"induction": {}, "random": {}}
    for ab_type in ["induction", "random"]:
        results[ab_type] = {}
        for p in percent_list:
            # 1) block_config
            mask = masks[ab_type][p]  # Tensor[num_layers, num_heads]
            block_config = {
                layer: [h for h, v in enumerate(mask[layer].tolist()) if v == 0.0]
                for layer in range(num_layers)
            }

            # 2) Disable Wo_h
            try:
                orig_blocks = disable_Wo_heads(model, block_config)
            except Exception as e:
                print(f"[ERROR] disable_Wo_heads failed at p={p}: {e}")
                orig_blocks = {}

            # 3) Generate with ablation
            ablated = []
            try:
                for tc in test_dataset:
                    prompt = tc["prompt"]
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out_ids = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            temperature=1.0,       
                            top_p=1.0,
                        )
                    ablated.append(tokenizer.decode(out_ids[0], skip_special_tokens=True))
            except Exception as e:
                print(f"[ERROR] generation failed at p={p}: {e}")

            # 4) Restore Wo_h
            try:
                restore_Wo_heads(model, orig_blocks)
            except Exception as e:
                print(f"[ERROR] restore_Wo_heads failed at p={p}: {e}")

            results[ab_type][p] = ablated

    return results


############Analyze results########################

def compute_head_ablation_accuracy(hd_result):
    accuracies = {}
    for task, res in hd_result.items():
        gts     = res['ground_truths']
        nor_p   = res['nor_']
        n       = len(gts)
        # baseline (no ablation) accuracy
        nor_acc = sum(1 for gt, p in zip(gts, nor_p) if gt == p) / n

        hd_acc = {}
        for mode, counts_map in res['hd'].items():
            mode_acc = {}
            for k, preds in counts_map.items():
                # pair up ground truth and preds
                length = min(n, len(preds))
                correct = sum(1 for i in range(length) if gts[i] == preds[i])
                mode_acc[k] = correct / length if length > 0 else float('nan')
            hd_acc[mode] = mode_acc

        accuracies[task] = {
            'nor_acc': nor_acc,
            'hd_acc':  hd_acc
        }

    return accuracies

def plot_head_ablation_results(accs):
    tasks    = list(accs.keys())
    k_values = sorted(accs[tasks[0]]['hd_acc']['induction'].keys())
    xs       = [0] + k_values   # prepend 0 for the baseline

    fig, axes = plt.subplots(1, len(tasks), sharey=True, figsize=(len(tasks)*2, 3))
    for ax, task in zip(axes, tasks):
        data = accs[task]
        nor  = data['nor_acc']
        ind  = [data['hd_acc']['induction'][k] for k in k_values]
        rnd  = [data['hd_acc']['random']   [k] for k in k_values]

        # prepend baseline
        ind_plot = [nor] + ind
        rnd_plot = [nor] + rnd
        ax.set_box_aspect(1)
        ax.plot(xs, ind_plot, 'o-',  label='Induction', markersize=4)
        ax.plot(xs, rnd_plot, 's--', label='Random',    markersize=4)

        ax.set_title(task, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xlabel("Percentage of Heads Ablated", fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)


    axes[0].set_ylabel("Accuracy", fontsize=9)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=2, frameon=True,
        fontsize=8, title="Ablation Type", title_fontsize=9,
        bbox_to_anchor=(0.5, -0.08)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()