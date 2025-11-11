import pandas as pd
import torch
from tqdm.notebook import tqdm
import json
import os
import repetition_neurons
import induction_heads
import functools
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer,  GenerationConfig

def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def load_data(path_to_data):
    tmp = [json.loads(line) for line in open(path_to_data)]
    return tmp

def load_model(model_name='meta-llama/Llama-3.1-8B',
    token=None,
    seed = 42):

    seed_everything(seed)

    login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, attn_implementation="eager"
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer
def generate_answer(model, tokenizer, dataset, seed):
    seed_everything(seed)
    device = model.device
    answers = []
    for seq in tqdm(dataset,desc="Generate answer by normal mode"):
        prompt = seq['prompt']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,       # Deterministic decoding
                top_p = None,
                temperature = None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answers.append(generated_text)
    return answers

def extract_answer(text, shots):
    lines = text.split('\n')
    if shots == 5:
        return lines[7] if len(lines) > 7 else ""
    else:
        return  lines[12] if len(lines) > 12 else "" #lines[12:]' '.join(lines[12:])
    
def extract_answers_nor(nor_answers, shots = 5):
    results =[]
    for text in nor_answers:
        results.append(extract_answer(text, shots)[-3:]) #modify extract class only
    return results


def extract_answers_ab(ablation_results, shots = 5):
    '''Extract answer after ablation both repetition neurons(all layers) and inducition heads'''
    extracted_answers = {}
    for ablation_type, percent_dict in ablation_results.items():
        extracted_answers[ablation_type] = {}
        for percent, outputs in percent_dict.items():
                extracted_answers[ablation_type][percent]= [extract_answer(output, shots)[-3:] for output in outputs]
    return extracted_answers

def extract_answers_ab_layer(all_results, shots=5):
    '''Extract answer after ablation repetiton neurons across layer segments'''
    ablation_results = {}
    if all_results:
        for mode, mode_results in all_results.items():
            ablation_results[mode] = {}
            for neurons_to_ablate, segment_results in mode_results.items():
                ablation_results[mode][neurons_to_ablate] = {}
                for segment_result in segment_results:
                    segment_range = segment_result["segment"]
                    generated_texts = segment_result["results"]
                    key = f"{segment_range.replace('->', '_')}" #ab_layer_{mode}_{neurons_to_ablate}_
                    extracted_list = []
                    for text in generated_texts:
                        extracted_list.append(extract_answer(text, shots)[-3:])#modify extract here
                    ablation_results[mode][neurons_to_ablate][key] = extracted_list
    return ablation_results


def analyze_class_overview(test_dataset, classes=["Foo", "Bar"]):
    """
    Analyze the performance of the model for each class (e.g., 'Foo', 'Bar') and visualize as tables.

    Parameters:
    - test_dataset: List of dictionaries representing the dataset.
    - classes: List of classes to analyze (e.g., ["Foo", "Bar"]).

    Returns:
    - overview: Dictionary containing counts, accuracy, and error analysis for each class.
    """
    df = pd.DataFrame(test_dataset)
    overview = {}

    for cls in classes:
        df_filtered = df[df['ground_truth'] == cls]
        total_rows = len(df_filtered)

        if total_rows == 0:
            print(f"No rows found for class '{cls}'. Skipping...")
            continue
        correct_counts = df_filtered.apply(lambda col: col.astype(str).str.contains(f": {cls}", na=False).sum())
        other_class = "Bar" if cls == "Foo" else "Foo"
        error_counts = df_filtered.apply(lambda col: col.astype(str).str.contains(f": {other_class}", na=False).sum())
        accuracy = correct_counts / total_rows
        overview[cls] = {
            "total_rows": total_rows,
            #"correct_counts": correct_counts.to_dict(),
            #"error_counts": error_counts.to_dict(),
            "accuracy": accuracy.to_dict()
        }

    return overview

def split_results_by_mode(overview_results):
    """
    Split the overview results into 'top' and 'random' modes based on key prefixes.

    Parameters:
    - overview_results: Dictionary containing the results of the ablation study.

    Returns:
    - overview_results_top: Dictionary containing only 'top' mode results.
    - overview_results_random: Dictionary containing only 'random' mode results.
    """
    overview_results_top = {}
    overview_results_random = {}

    for cls, metrics in overview_results.items():
        accuracy_top = {k: v for k, v in metrics["accuracy"].items() if 'top' in k or k == "nor"}
        accuracy_random = {k: v for k, v in metrics["accuracy"].items() if 'random' in k or k == "nor"}
        overview_results_top[cls] = {
            "total_rows": metrics["total_rows"],
            #"correct_counts": metrics["correct_counts"],
            #"error_counts": metrics["error_counts"],
            "accuracy": accuracy_top,
        }
        overview_results_random[cls] = {
            "total_rows": metrics["total_rows"],
            #"correct_counts": metrics["correct_counts"],
            #"error_counts": metrics["error_counts"],
            "accuracy": accuracy_random,
        }

    return overview_results_top, overview_results_random


def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        if isinstance(data, dict):
            for key, value in data.items():
                json_line = {key: value}
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        elif isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            raise ValueError("Unsupported data type. Data must be a dictionary or list.")

def save_experiment_results(nor_answer, 
                            ab_nr_all_result, 
                            ab_nr_layer_result, 
                            ab_hd_result, 
                            file_paths, 
                            task_name, 
                            model_name, 
                            shots):
    if model_name == 'Qwen/Qwen2.5-7B':
        model_subdir = 'qwen25_7b'
    elif model_name == 'meta-llama/Llama-3.1-8B':
        model_subdir = 'llama31_8b'
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # Construct the full save path
    BASE_DIR = os.getcwd()
    dict_path = os.path.join(BASE_DIR, 'data', 'ab_result', model_subdir, f"{shots}_shot")
    os.makedirs(dict_path, exist_ok=True)
    full_file_paths = {key: os.path.join(dict_path, f"{task_name}_{value}") for key, value in file_paths.items()}
    save_to_jsonl(nor_answer, full_file_paths["nor_answer"])
    save_to_jsonl(ab_nr_all_result, full_file_paths["ab_nr_all_result"])
    save_to_jsonl(ab_nr_layer_result, full_file_paths["ab_nr_layer_result"])
    save_to_jsonl(ab_hd_result, full_file_paths["ab_hd_result"])


def compute_ablation_accuracies(results_dict, mode='top', K=250):
    def class_accuracy(gt_list, pred_list, cls):
        idxs = [i for i, gt in enumerate(gt_list) if gt == cls]
        if not idxs:
            return float('nan')
        correct = sum(1 for i in idxs if pred_list[i] == cls)
        return correct / len(idxs)

    accuracies = {}
    for task, res in results_dict.items():
        gts = res['ground_truths']
        try:
            seg_preds = res['nr_layer'][mode][K]
        except KeyError:
            raise KeyError(f"No data for task={task}, mode={mode}, K={K}")

        task_acc = {}
        for segment, preds in seg_preds.items():
            foo_acc = class_accuracy(gts, preds, 'Foo')
            bar_acc = class_accuracy(gts, preds, 'Bar')
            task_acc[segment] = {'Foo': foo_acc, 'Bar': bar_acc}
        task_acc['Foo_nor'] = class_accuracy(gts, res['nor_'], 'Foo')
        task_acc['Bar_nor'] = class_accuracy(gts, res['nor_'], 'Bar')

        accuracies[task] = task_acc

    return accuracies



def extract_deactivated_neurons(all_results):
    """
    Extract deactivated neurons from the ablation study results.

    Parameters:
    - all_results: Dictionary containing the results of the ablation study.

    Returns:
    - deactivated_neurons: Dictionary mapping keys (e.g., "20_0_0.2") to lists of deactivated neurons.
    """
    deactivated_neurons = {}

    for neurons_to_ablate, segment_results in all_results.items():
        for segment_result in segment_results:
            segment_range = segment_result["segment"]  # e.g., "0->0.2"
            neurons_ablated = segment_result["neurons_ablated"]  # List of (layer_idx, neuron_idx)
            key = f"{neurons_to_ablate}_{segment_range.replace('->', '_')}"
            deactivated_neurons[key] = neurons_ablated

    return deactivated_neurons

def analyze_diffs_of_deactivated_neurons(deactivated_neurons, sortedNeurons):
    """
    Analyze the 'diffs' values of deactivated neurons.

    Parameters:
    - deactivated_neurons: Dictionary mapping keys (e.g., "20_0_0.2") to lists of deactivated neurons.
                           Each entry contains a list of (layer_idx, neuron_idx) tuples.
    - sortedNeurons: List of repetition neurons sorted by 'diffs'.
                     Each entry is a dictionary with keys like 'neuron' and 'diffs'.

    Returns:
    - diffs_by_group: Dictionary containing 'diffs' values grouped by segment and number of neurons.
                      Keys are formatted as "neurons_to_ablate_segment_range" (e.g., "20_0_0.2").
    """
    diffs_by_group = {}

    for key, neurons_ablated in deactivated_neurons.items():
        diffs = []
        for layer_idx, neuron_idx in neurons_ablated:
            for neuron_info in sortedNeurons:
                if neuron_info["neuron"] == (layer_idx, neuron_idx):
                    diffs.append(neuron_info["diffs"])
                    break  
        diffs_by_group[key] = diffs

    return diffs_by_group



def conduct_segment_dual_ablation_2phase(model, tokenizer, dataset, neurons_by_layer_position, 
                                         segment_ranges, neurons_to_ablate, modeNr="top",
                                         head_mask=None, shot=10, seed=42):
    seed_everything(seed)
    texts = [{'ids': item['prompt']} for item in dataset]
    total_layers = len(model.model.layers)
    results = []
    for seg_range in segment_ranges:
        segment_neurons = []
        segment_layer_indices = set()
        for rel_pos, neurons in neurons_by_layer_position.items():
            if seg_range[0] <= rel_pos < seg_range[1]:
                segment_neurons.extend(neurons)
                segment_layer_indices.add(int(rel_pos * total_layers))

        if modeNr == "top":
            segment_neurons_sorted = sorted(segment_neurons, key=lambda x: x['diffs'], reverse=True)
            selected_neurons = [n['neuron'] for n in segment_neurons_sorted[:neurons_to_ablate]]
        elif modeNr == "random":
            import random
            random.shuffle(segment_neurons)
            selected_neurons = [n['neuron'] for n in segment_neurons[:neurons_to_ablate]]

        neurons_by_layer = {}
        for neuron in selected_neurons:
            layer_idx, neuron_idx = neuron
            neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)

        #attn_hooks = []
        #num_layers = model.config.num_hidden_layers
        orig_blocks = {}
        if head_mask is not None:
            block_config = {
                l: [h for h, v in enumerate(mask.tolist()) if v == 0.0]
                for l, mask in enumerate(head_mask) if (mask == 0).any()
            }
            orig_blocks = induction_heads.disable_Wo_heads(model, block_config)

        try:
            deactivators = []
            for layer_idx in segment_layer_indices:
                if layer_idx in neurons_by_layer:
                    deactivators.append(
                        repetition_neurons.Deactivator(
                            model.model.layers[layer_idx].mlp.act_fn,
                            neurons_by_layer[layer_idx],
                            "all"
                        )
                    )
            segment_results = []
            for text_dict in texts:
                inputs = tokenizer(text_dict["ids"], return_tensors="pt").to(model.device)
                outputs = model.generate( 
                    **inputs,
                    generation_config=GenerationConfig(
                        max_new_tokens=1,
                        do_sample=False,
                        eos_token_id=model.config.eos_token_id,
                        pad_token_id=model.config.eos_token_id,
                    ),
                )
                seg_ans = extract_answer(tokenizer.decode(outputs[0], skip_special_tokens=True), shot)
                segment_results.append(seg_ans[-3:])

        finally:

            if head_mask is not None and orig_blocks:
                induction_heads.restore_Wo_heads(model, orig_blocks)
            for d in deactivators:
                d.release()
        results.append({
            "segment": f"{seg_range[0]}->{seg_range[1]}",
            #"neurons_ablated": selected_neurons,
            "results": segment_results
        })
    return results
def run_dual_ablation_study(model, tokenizer, dataset, sortedNeurons, id_avg_scores, 
                              percent_list=[1, 3], segment_ranges=[(0, 0.2)], 
                              neurons_to_ablate_list=[20, 50], shot=10, seed=42):
    seed_everything(seed)
    num_layers = len(model.model.layers)
    num_heads = id_avg_scores[0].shape[0]
    n_heads = model.config.num_attention_heads
    device = model.device

    # Build a base causal mask for attention hooks.
    global base_causal_mask
    max_seq_len = 1024
    base_causal_mask = torch.tril(torch.ones((1, n_heads, max_seq_len, max_seq_len), dtype=torch.uint8)).to(device)
    masks = induction_heads.build_head_masks(id_avg_scores, num_layers, num_heads, 
                                              percent_list=percent_list, random_seed=seed)
    neurons_by_layer_position = {}
    for neuron_info in sortedNeurons:
        layer_idx, _ = neuron_info['neuron']
        rel = layer_idx / float(num_layers)
        neurons_by_layer_position.setdefault(rel, []).append(neuron_info)

    results = {}
    rep_modes = ["top", "random"]      # for repetition neuron selection
    mask_modes = ["induction", "random"]  # for head mask selection

    for rep_mode in rep_modes:
        results[rep_mode] = {}
        for mask_mode in tqdm(mask_modes, desc=f"{rep_mode} Neurons"):
            results[rep_mode][mask_mode] = {}
            for neurons_to_ablate in neurons_to_ablate_list:
                results[rep_mode][mask_mode][neurons_to_ablate] = {}
                for p in percent_list:
                    head_mask = masks[mask_mode][p] if mask_mode in masks and p in masks[mask_mode] else None

                    res = conduct_segment_dual_ablation_2phase(
                        model, tokenizer, dataset,
                        neurons_by_layer_position,
                        segment_ranges,
                        neurons_to_ablate,
                        modeNr=rep_mode,
                        head_mask=head_mask,
                        shot=shot,
                        seed=seed
                    )
                    results[rep_mode][mask_mode][neurons_to_ablate][p] = res
    return results