import pandas as pd
import torch
from tqdm.notebook import tqdm
import json
import os
from .  import repetition_neurons
from . import  induction_heads
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
    with open(path_to_data, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(model_name='meta-llama/Llama-3.1-8B',
    token=None,
    seed = 42):

    seed_everything(seed)

    hf_token = token or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, attn_implementation="eager"
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer
def generate_answer(model, tokenizer, dataset, seed, task='abstract', rep_neuron= None,):
    seed_everything(seed)
    assert task in ['abstract', 'wmt']
    device = model.device
    answers = []
    probe = None
    if rep_neuron is not None:
        by_layer = repetition_neurons._neurons_to_by_layer(rep_neuron, len(model.model.layers))
        if by_layer:
            probe = repetition_neurons.NeuronMeanProbe(model, by_layer)
            probe.attach()
    try:
        for seq in tqdm(dataset, desc="Generate answer by normal mode"):
            prompt = seq['prompt']
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            if task == 'abstract':
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,       # Deterministic decoding
                        top_p=None,
                        temperature=None,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated_text = tokenizer.decode(
                output_ids[0], skip_special_tokens=True)
            elif task == 'wmt':
                last_source = prompt.rsplit("Source:", 1)[-1]
                # Tokenize just that segment to measure its length
                src_ids = tokenizer(last_source, return_tensors="pt")["input_ids"]
                src_len = src_ids.shape[-1]

                # Compute max_new_tokens dynamically
                max_new = int(src_len * 1.2)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=False,       # Deterministic decoding
                        top_p=None,
                        temperature=None,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id
                    )
                input_len = inputs["input_ids"].shape[-1]
                new_ids = output_ids[0][input_len:]
                generated_text = tokenizer.decode(
                    new_ids, skip_special_tokens=True)
            answers.append(generated_text)
    finally:
        if probe is not None:
            probe.release()
    if probe is not None:
        return answers, probe.mean_all()#probe.means_jsonable()
    else:
        return answers, None

def extract_answer(text):
    lines = text.split('\n')
    return lines[-1]
   
def extract_answers_nor(nor_answers):
    results = []
    for text in nor_answers:
        results.append(extract_answer(text).split(": ")[-1])
    return results
def extract_answers_nor_sst2(nor_answers, shots=5):
    results = []
    for text in nor_answers:
        results.append(extract_answer(text).split(": ")[-1])
    return results

def extract_answers_nor_wmt(nor_answers):
    results = []
    for text in nor_answers:
        text = text.strip()
        results.append(text.split("\n\n")[0])
    return results

def extract_answers_ab(ablation_results, task: str ='abstract'):
    '''Extract answer after ablation inducition heads'''
    assert task in ['abstract', 'wmt']
    extracted_answers = {}
    for ablation_type, percent_dict in ablation_results.items():
        extracted_answers[ablation_type] = {}
        for percent, outputs in percent_dict.items():
            if task =='abstract':
                extracted_answers[ablation_type][percent] = [extract_answer(
                    output).split(": ")[-1] for output in outputs]
            elif task =='wmt':
                extracted_answers[ablation_type][percent] = [output.strip().split("\n\n")[0] for output in outputs]
    return extracted_answers

def extract_answers_ab_layer(all_results, task='abstract'):
    '''Extract answer after ablation repetiton neurons across layer segments'''
    assert task in ['abstract', 'wmt']
    ablation_results = {}
    if all_results:
        for mode, mode_results in all_results.items():
            ablation_results[mode] = {}
            for neurons_to_ablate, segment_results in mode_results.items():
                ablation_results[mode][neurons_to_ablate] = {}
                for segment_result in segment_results:
                    segment_range = segment_result["segment"]
                    generated_texts = segment_result["results"]
                    # ab_layer_{mode}_{neurons_to_ablate}_
                    key = f"{segment_range.replace('->', '_')}"
                    extracted_list = []
                    for text in generated_texts:
                        # extracted_list.append(extract_answer(text, shots)[-3:])#modify extract here
                        if task =='abstract':
                            ex_rs = extract_answer(text).split(": ")[-1]
                        elif task =='wmt':
                            ex_rs = text.strip().split("\n\n")[0]
                        extracted_list.append(ex_rs)
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
            "accuracy": accuracy_top,
        }
        overview_results_random[cls] = {
            "total_rows": metrics["total_rows"],
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
def analyze_exp_1(ab_dataset, nor_, nr_all, hd):
    '''Get results from first experiment, include normal answer, ablation repetition neurons and induction heads'''
    ground_truths = [item['ground_truth'] for item in ab_dataset]
    queries = [item['query'] for item in ab_dataset]
    df = pd.DataFrame()
    df['query'] = queries
    df['ground_truth'] = ground_truths
    df['nor'] = nor_
    for mode, mode_results in nr_all.items():
        for n_nr_ab, results in mode_results.items():
            cl_name = f'ab_nr_{mode}_{n_nr_ab}'
            df[cl_name] = results
    for mode, mode_results in hd.items():
        if mode == 'induction':
            mode = 'top'
        for n_nr_ab, results in mode_results.items():
            cl_name = f'ab_id_{mode}_{n_nr_ab}'
            df[cl_name] = results
    return df


def analyze_exp_2(ab_dataset, nor_, nr_layer):
    ground_truths = [item['ground_truth'] for item in ab_dataset]
    queries = [item['query'] for item in ab_dataset]
    df = pd.DataFrame()
    df['query'] = queries
    df['ground_truth'] = ground_truths
    df['nor'] = nor_
    for mode, mode_results in nr_layer.items():
        for n_nr_layer, segment_results in mode_results.items():
            for segment, segment_result in segment_results.items():
                cl_name = f"ab_ly_{mode}_{n_nr_layer}_{segment.replace('->', '_')}"
                df[cl_name] = segment_result

    return df

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


def compute_dual_ablation_accuracies_by_class(joint_results, classes=None):
    """
    From joint_results (with keys 'ground_truths', 'nor_', 'ab_dual'),
    compute per-task baseline and per-segment class accuracies.
    """
    def normalize_label(pred):
        if pred is None:
            return ""
        text = str(pred).strip()
        if ": " in text:
            text = text.split(": ")[-1].strip()
        if not text:
            return ""
        token = text.split()[0].strip(".,;:!?\"'")
        return token

    out = {}
    for task, info in joint_results.items():
        gts = info["ground_truths"]

        task_classes = classes if classes is not None else sorted(set(gts))
        class_idxs = {
            cls: [i for i, gt in enumerate(gts) if gt == cls]
            for cls in task_classes
        }

        nor_preds = [normalize_label(p) for p in info["nor_"]]
        baseline = {}
        for cls, idxs in class_idxs.items():
            if idxs:
                baseline[cls] = sum(1 for i in idxs if nor_preds[i] == cls) / len(idxs)
            else:
                baseline[cls] = float("nan")

        ablation = {}
        for rep_mode, rep_block in info["ab_dual"].items():
            ablation[rep_mode] = {}
            for mask_mode, mask_block in rep_block.items():
                ablation[rep_mode][mask_mode] = {}
                for k, pct_block in mask_block.items():
                    ablation[rep_mode][mask_mode][k] = {}
                    for pct, seg_list in pct_block.items():
                        seg_map = {}
                        for seg_res in seg_list:
                            seg = seg_res["segment"]
                            preds = [normalize_label(p) for p in seg_res["results"]]
                            cls_acc = {}
                            for cls, idxs in class_idxs.items():
                                if idxs:
                                    cls_acc[cls] = sum(
                                        1 for i in idxs if preds[i] == cls
                                    ) / len(idxs)
                                else:
                                    cls_acc[cls] = float("nan")
                            seg_map[seg] = cls_acc
                        ablation[rep_mode][mask_mode][k][pct] = seg_map

        out[task] = {
            "baseline": baseline,
            "ablation": ablation,
        }

    return out



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
                                         head_mask=None, shot=10, seed=42, task: str ='abstract'):
    assert task in ['abstract', 'wmt']
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
            segment_neurons_sorted = sorted(
                segment_neurons, key=lambda x: x['diffs'], reverse=True)
            selected_neurons = [n['neuron']
                                for n in segment_neurons_sorted[:neurons_to_ablate]]
        elif modeNr == "random":
            import random
            random.shuffle(segment_neurons)
            selected_neurons = [n['neuron']
                                for n in segment_neurons[:neurons_to_ablate]]

        neurons_by_layer = {}
        for neuron in selected_neurons:
            layer_idx, neuron_idx = neuron
            neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)

        # attn_hooks = []
        # num_layers = model.config.num_hidden_layers
        orig_blocks = {}
        if head_mask is not None:
            block_config = {
                l: [h for h, v in enumerate(mask.tolist()) if v == 0.0]
                for l, mask in enumerate(head_mask) if (mask == 0).any()
            }
            orig_blocks = induction_heads.disable_Wo_heads(model, block_config)

        try:
            # ablate repetition neurons
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

            # GENERATE
            segment_results = []
            for text_dict in texts:
                inputs = tokenizer(
                    text_dict["ids"], return_tensors="pt").to(model.device)
                if task =='abstract':
                    max_tok = 1
                elif task =='wmt':
                    last_source = text_dict["ids"].rsplit("Source:", 1)[-1]
                    src_ids = tokenizer(last_source, return_tensors="pt")["input_ids"]
                    src_len = src_ids.shape[-1]
                    max_tok = src_len*1.1
                outputs = model.generate(
                    **inputs,
                    generation_config=GenerationConfig(
                        max_new_tokens=max_tok,
                        do_sample=False,
                        eos_token_id=model.config.eos_token_id,
                        pad_token_id=model.config.eos_token_id,
                    ),
                )
                input_len = inputs["input_ids"].shape[-1]
                new_ids = outputs[0][input_len:]
                decoded_text = tokenizer.decode(
                    new_ids, skip_special_tokens=True)
                if task =='abstract':
                    seg_ans = extract_answer(decoded_text).split(": ")[-1]
                elif task =='wmt':
                    seg_ans = decoded_text.strip().split("\n\n")[0]
                #segment_results.append(seg_ans.split(": ")[-1])
                segment_results.append(seg_ans)

        finally:

            if head_mask is not None and orig_blocks:
                induction_heads.restore_Wo_heads(model, orig_blocks)
            for d in deactivators:
                d.release()
        results.append({
            "segment": f"{seg_range[0]}->{seg_range[1]}",
            # "neurons_ablated": selected_neurons,
            "results": segment_results
        })
    return results


def run_dual_ablation_study(
    model,
    tokenizer,
    dataset,
    sortedNeurons,
    id_avg_scores,
    percent_list=[1, 3],
    segment_ranges=[(0, 0.2)],
    neurons_to_ablate_list=[20, 50],
    shot=10,
    seed=42,
    task: str = 'abstract',
    dataset_name: str | None = None,
    progress: str = "full",
):
    assert task in ['abstract','wmt']
    seed_everything(seed)
    num_layers = len(model.model.layers)
    num_heads = id_avg_scores[0].shape[0]
    n_heads = model.config.num_attention_heads
    device = model.device

    # Build a base causal mask for attention hooks.
    global base_causal_mask
    max_seq_len = 1024
    base_causal_mask = torch.tril(torch.ones(
        (1, n_heads, max_seq_len, max_seq_len), dtype=torch.uint8)).to(device)

    # Build head masks using your build_head_masks function.
    # This returns a dictionary with keys: "induction" and "random"
    masks = induction_heads.build_head_masks(id_avg_scores, num_layers, num_heads,
                                             percent_list=percent_list, random_seed=seed)

    # Group sortedNeurons by their relative layer position.
    neurons_by_layer_position = {}
    for neuron_info in sortedNeurons:
        layer_idx, _ = neuron_info['neuron']
        rel = layer_idx / float(num_layers)
        neurons_by_layer_position.setdefault(rel, []).append(neuron_info)

    results = {}
    rep_modes = ["top", "random"]      # for repetition neuron selection
    mask_modes = ["induction", "random"]  # for head mask selection

    label = dataset_name or task
    for rep_mode in rep_modes:
        results[rep_mode] = {}
        show_mask = progress in ("full", "outer")
        mask_iter = tqdm(
            mask_modes,
            desc=f"{label} | {rep_mode} neurons",
            leave=False,
            disable=not show_mask,
        )
        for mask_mode in mask_iter:
            results[rep_mode][mask_mode] = {}
            show_neurons = progress == "full"
            neuron_iter = tqdm(
                neurons_to_ablate_list,
                desc=f"{label} | {rep_mode}/{mask_mode}",
                leave=False,
                disable=not show_neurons,
            )
            for neurons_to_ablate in neuron_iter:
                # print(f"ablate {neurons_to_ablate} neurons")
                results[rep_mode][mask_mode][neurons_to_ablate] = {}
                show_percent = progress == "full"
                percent_iter = tqdm(
                    percent_list,
                    #desc=f"{label} | {rep_mode}/{mask_mode}/{neurons_to_ablate}",
                    leave=False,
                    disable=not show_percent,
                )
                for p in percent_iter:
                    head_mask = masks[mask_mode][p] if mask_mode in masks and p in masks[mask_mode] else None

                    res = conduct_segment_dual_ablation_2phase(
                        model, tokenizer, dataset,
                        neurons_by_layer_position,
                        segment_ranges,
                        neurons_to_ablate,
                        modeNr=rep_mode,
                        head_mask=head_mask,
                        shot=shot,
                        seed=seed,
                        task=task
                    )
                    results[rep_mode][mask_mode][neurons_to_ablate][p] = res
    return results
