import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import random
import json
import re
from tqdm import tqdm
from huggingface_hub import login
import functools


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


    

class OutputInspector:
        def __init__(self, target_layer):
            self.layer_outputs = []
            self.feature_handle = target_layer.register_forward_hook(self.feature)

        def feature(self, model, input, output):
            self.layer_outputs.append(output.detach().cpu())

        def release(self):
            self.feature_handle.remove()

def count_overlap(text, query):
    return len(re.findall(query, text, overlapped=True))

def get_first_appearing_idx(text, query):
    return re.finditer(query, text).__next__().start(0)

def detect_repetition(line, n, r, k):
    SEP = ' '
    for i in range(0, len(line) - n + 1):
        ngram = line[i:i + n]
        ngram_str = SEP.join(map(str, ngram))
        line_range = line[max(0, i + n - r):i + n]
        line_range_str = SEP.join(map(str, line_range))
        count_rep_in_range = count_overlap(line_range_str, ngram_str)

        if k <= count_rep_in_range:
            try:
                first_position = get_first_appearing_idx(line_range_str, ngram_str)
                first_position = len(line_range_str[:first_position].split())
                first_position += max(0, i + n - r)

                line_range_str4second = SEP.join(map(str, line[first_position + 1:i + n]))
                second_position = get_first_appearing_idx(line_range_str4second, ngram_str)
                second_position = len(line_range_str4second[:second_position].split())
                second_position += first_position + 1

                line_range_str4third = SEP.join(map(str, line[second_position + 1:i + n]))
                third_position = get_first_appearing_idx(line_range_str4third, ngram_str)
                third_position = len(line_range_str4third[:third_position].split())
                third_position += second_position + 1

                if (second_position - first_position) == (third_position - second_position):
                    return ngram, first_position, second_position, third_position
            except:
                pass
    return [], -1, -1, -1

def get_acts(model, tokenizer, input_ids):
    model.eval()
    with torch.no_grad():
        if 'GemmaForCausalLM' in str(type(model)) or 'LlamaForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
            act_inspectors = [OutputInspector(layer.mlp.act_fn) for layer in model.model.layers]
        elif 'GPTNeoXForCausalLM' in str(type(model)):
            act_inspectors = [OutputInspector(layer.mlp.act) for layer in model.gpt_neox.layers]
        elif 'PhiForCausalLM' in str(type(model)):
            act_inspectors = [OutputInspector(layer.mlp.activation_fn) for layer in model.model.layers]
        else:
            print('Model is not supported!')

        input_ids = torch.LongTensor([input_ids]).to(model.device)
        outputs = model(input_ids)

        for act_inspector in act_inspectors:
            act_inspector.release()

        acts = torch.cat([torch.cat(act_inspector.layer_outputs, dim=1) for act_inspector in act_inspectors], dim=0).transpose(0, 1)
    return acts

def get_averaged_activations(data, model, tokenizer, max_range, position):
    rep_position = f'{position}Position'

    normal_acts = None
    repeti_acts = None
    normal_total_points = 0
    repeti_total_points = 0

    for line in tqdm(data, desc="Compute activation", unit="sequence"):
        input_ids = line['generatedIds']
        acts = get_acts(model, tokenizer, input_ids)
        starting_point = line[rep_position] - 1
        normal_range = list(range(max(0, starting_point - max_range), starting_point))
        repeti_range = list(range(starting_point, min(len(input_ids), starting_point + max_range)))

        normal_total_points += len(normal_range)
        repeti_total_points += len(repeti_range)

        na = acts[normal_range].sum(dim=0)
        ra = acts[repeti_range].sum(dim=0)

        if normal_acts is None:
            normal_acts = na
        else:
            normal_acts += na

        if repeti_acts is None:
            repeti_acts = ra
        else:
            repeti_acts += ra

    normal_acts /= normal_total_points
    repeti_acts /= repeti_total_points

    return normal_acts, repeti_acts


def find_neurons(data, model, tokenizer, max_range, position):
        normal_acts, repeti_acts = get_averaged_activations(data, model, tokenizer, max_range, position)
        diff = repeti_acts - normal_acts
        ranks = torch.argsort(diff.flatten(), descending=True)
        width = diff.shape[1]
        sorted_neurons = []
        for r in ranks:
            neuron = (int(r // width), int(r % width))
            info = {
                'neuron': neuron,
                'normalActs': normal_acts[neuron].tolist(),
                'repetitionActs': repeti_acts[neuron].tolist(),
                'diffs': diff[neuron].tolist()
            }
            sorted_neurons.append(info)
        return sorted_neurons


def find_sorted_neurons(model,
    tokenizer,
    base_data,
    max_range=30,
    position='second',
    seed=42
):
    print("Sorted Repetition Neurons")
    
    seed_everything(seed)

    # repetition_dataset = [json.loads(line) for line in open(path_to_data)]
    # rep1 = repetition_dataset[:1000] 

    sorted_neurons = find_neurons(base_data, model, tokenizer,  max_range=max_range, position=position,)
    return sorted_neurons


class Deactivator():
    def __init__(self, targetLayer, neuronIds, mode, lastN=0):
        self.neuronIds = neuronIds

        assert mode in ['last', 'all', 'lastN'], 'mode should be last or all'
        self.mode = mode
        self.lastN = lastN

        self.outputHandle = targetLayer.register_forward_hook(self.deactivate)

    def deactivate(self,model, input, output):
        if self.mode == 'last':
          output[0, -1, self.neuronIds] *= 0
        elif self.mode == 'all':
          output[0, :, self.neuronIds] *= 0
        elif self.mode == 'lastN':
          output[0, -self.lastN:, self.neuronIds] *= 0
        else:
          print(f'{self.mode=} cannot be recognized')
          pass
        return output

    def release(self):
        self.outputHandle.remove()

def convertNeuronsToDict(neurons):
    layer2neurons = {}
    for fn in neurons:
        i, j = fn
        if i not in layer2neurons:
            layer2neurons[i] = []
        layer2neurons[i].append(j)
    return layer2neurons


from tqdm import tqdm
import random
def conductExpIntervention(model, tokenizer, texts, neurons, mode, selectMode, K, N=50):
    assert mode in ['activate', 'deactivate'], 'mode should be activate or deactivate'
    assert selectMode in ['top', 'random'], 'selectMode should be top or random'

    numRep = 0

    logs = []

    if selectMode=='top':
        targetNeurons = [neuron['neuron'] for neuron in neurons[:K]]
    elif selectMode=='random':
        targetNeurons = [neuron['neuron'] for neuron in random.sample(neurons, K)]



    for i, text in enumerate(texts):
        if mode=='deactivate':
            #ngram, firstPosition, secondPosition, thirdPosition = detectRepetition(text['ids'], n=10, r=100, k=3)
            initialInput = text['ids']#[:secondPosition]
        elif mode=='activate':
            initialInput = text['ids'][:N]

        gen_answer = generateWithIntervention(model, tokenizer, initialInput, targetNeurons, mode=mode)

        logs.append(gen_answer)
    
    return logs#, numRep

def run_segment_ablation_study_2phase(model, tokenizer, texts, sortedNeurons, 
                                      segment_ranges =None, neurons_to_ablate_list=None, seed=42):
    """
    Run segment-wise ablation studies for both 'top' and 'random' modes.

    Parameters:
    - model: Large language model.
    - tokenizer: Tokenizer corresponding to the model.
    - texts: List of prompts or inputs to evaluate ICL.
    - sortedNeurons: List of repetition neurons in the format [{'neuron': (layer_idx, neuron_idx), 'normalActs': ..., 'repetitionActs': ..., 'diffs': ...}, ...].
    - segment_ranges: List of tuples representing the ranges of segments (e.g., [(0, 0.2), (0.4, 0.6), (0.8, 1.0)]).
    - neurons_to_ablate_list: List of neurons to ablate per segment (e.g., [20, 50, 100]).

    Returns:
    - all_results: Aggregated results for both 'top' and 'random' modes.
    """
    seed_everything(seed)
    if segment_ranges is None:
        segment_ranges = [(0, 0.2), (0.4, 0.6), (0.8, 1.0)]

    if neurons_to_ablate_list is None:
        neurons_to_ablate_list = [20, 50, 100, 180, 250]

    neurons_by_layer_position = {}
    total_layers = len(model.model.layers) 
    for neuron_info in sortedNeurons:
        layer_idx, neuron_idx = neuron_info['neuron']
        relative_position = layer_idx / total_layers  
        if relative_position not in neurons_by_layer_position:
            neurons_by_layer_position[relative_position] = []
        neurons_by_layer_position[relative_position].append(neuron_info)


    all_results = {"top": {}, "random": {}}


    for mode in ["top", "random"]:
        #print(f"\nRunning experiments in '{mode}' mode...")
        for neurons_to_ablate in neurons_to_ablate_list:
            #print(f"\nAblating {neurons_to_ablate} neurons per segment in '{mode}' mode...")
            results = conduct_segment_ablation_2phase(
                model, tokenizer, texts, 
                neurons_by_layer_position, segment_ranges, 
                neurons_to_ablate, mode=mode, seed=seed
            )
            all_results[mode][neurons_to_ablate] = results

    return all_results


def conduct_segment_ablation_2phase(model, tokenizer, dataset, neurons_by_layer_position, 
                                    segment_ranges, neurons_to_ablate, mode="top", seed=42):
    """
    Perform ablation by deactivating a specific number of neurons in each segment of the model.

    Parameters:
    - model: Large language model (e.g., LLaMA, GPT-NeoX).
    - tokenizer: Tokenizer corresponding to the model.
    - texts: List of prompts or inputs to evaluate ICL.
    - neurons_by_layer_position: Dictionary containing repetition neurons grouped by relative layer position.
    - segment_ranges: List of tuples representing the ranges of segments (e.g., [(0, 0.2), (0.4, 0.6), (0.8, 1.0)]).
    - neurons_to_ablate: Number of neurons to deactivate per segment.
    - mode: Mode of neuron selection ("top" for top neurons by diffs, "random" for random neurons).

    Returns:
    - results: ICL results after ablating neurons in each segment.
    """
    #print("Ablation repetition neurons in each segment")
    seed_everything(seed)
    texts = [{'ids': line['prompt']} for line in dataset]

    results = []

    for i, (start, end) in enumerate(segment_ranges):#tqdm(enumerate(segment_ranges), desc="Processing segment ranges", unit="segment"):
        #print(f"\nProcessing segment {i+1}: {start} -> {end}...")
        segment_neurons = []
        segment_layer_indices = set()  
        for layer_position, neurons in neurons_by_layer_position.items():
            if start <= layer_position < end:
                segment_neurons.extend(neurons)
                segment_layer_indices.add(int(layer_position * len(model.model.layers)))  # Map relative position to layer index

        if mode == "top":
            segment_neurons_sorted = sorted(segment_neurons, key=lambda x: x['diffs'], reverse=True)
            neurons_to_ablate_list = [neuron['neuron'] for neuron in segment_neurons_sorted[:neurons_to_ablate]]
        elif mode == "random":
            # Randomly select neurons
            import random
            random.shuffle(segment_neurons)
            neurons_to_ablate_list = [neuron['neuron'] for neuron in segment_neurons[:neurons_to_ablate]]
        else:
            raise ValueError("Invalid mode. Choose 'top' or 'random'.")

        neurons_by_layer = {}
        for neuron in neurons_to_ablate_list:
            layer_idx, neuron_idx = neuron
            if layer_idx not in neurons_by_layer:
                neurons_by_layer[layer_idx] = []
            neurons_by_layer[layer_idx].append(neuron_idx)

        deactivators = [] 
        for layer_idx in segment_layer_indices:
            if layer_idx in neurons_by_layer:
                deactivator = Deactivator(model.model.layers[layer_idx].mlp.act_fn, neurons_by_layer[layer_idx], 'all')
                deactivators.append(deactivator)


        segment_results = []
        for text in texts:#tqdm(texts, desc=f"Segment {i+1}", unit="text"):
            text = text['ids']
            initialInput = tokenizer(text, return_tensors="pt")
            initialInput = initialInput["input_ids"].to(model.device)

            # Inference with intervention
            generationConfigGreedy = GenerationConfig(
                max_new_tokens=1,
                do_sample=False,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.eos_token_id,
                top_p=0,
                temperature=1.0, 
            )
            additionalOutputs = model.generate(initialInput, generation_config=generationConfigGreedy)

            # Decode output
            token_ids = additionalOutputs[0].tolist()
            decoded_text = tokenizer.decode(token_ids)
            segment_results.append(decoded_text)

        # Release all deactivators
        for deactivator in deactivators:
            deactivator.release()

        # Save results for the current segment
        results.append({
            "segment": f"{start}->{end}",
            "neurons_ablated": neurons_to_ablate_list,
            "results": segment_results
        })

    return results

def wrap_neuron_intervention_forward(original_forward, neuron_indices, mode='deactivate'):
    """
    Wraps the MLP forward function to intervene on selected neurons.

    Args:
        original_forward (Callable): The original forward function of the MLP module.
        neuron_indices (list): List of indices (within the hidden dimension) corresponding to the neurons to intervene.
        mode (str): Intervention mode. 'deactivate' zeros out the neurons; 'activate' sets them to a fixed value.

    Returns:
        Callable: The wrapped forward function.
    """
    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        # Get the output from the original forward pass.
        # Expected output shape: [batch_size, seq_length, hidden_dim]
        output = original_forward(*args, **kwargs)

        # Intervention on the selected neurons.
        if mode == 'deactivate':
            # Zero out activations for the chosen neuron indices.
            for neuron_idx in neuron_indices:
                output[..., neuron_idx] = 0.0
        elif mode == 'activate':
            # As an example, force the activation to a constant value (e.g., 1.0).
            for neuron_idx in neuron_indices:
                output[..., neuron_idx] = 1.0
        else:
            raise ValueError("Unsupported mode. Choose 'deactivate' or 'activate'.")
        
        return output

    return wrapped_forward


def set_neuron_intervention_hooks(model, target_neurons, mode='deactivate'):
    """
    Installs hooks on the model's MLP modules to intervene on selected repetition neurons.
    
    Args:
        model: Transformer model.
        target_neurons (list): List of tuples (layer_idx, neuron_idx) specifying the neurons to intervene.
        mode (str): Intervention mode. Options are 'deactivate' or 'activate'.
        
    Returns:
        hooks (list): List of tuples (layer_idx, original_forward) to allow removal later.
    """
    hooks = {}
    
    # Group the target neurons by their layer index.
    neurons_by_layer = {}
    for layer_idx, neuron_idx in target_neurons:
        neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)
    
    # Loop over each layer that has target neurons.
    # It is assumed that the MLP (or feedforward) module is located at model.model.layers[layer_idx].mlp.
    for layer_idx, neuron_list in neurons_by_layer.items():
        mlp_module = model.model.layers[layer_idx].mlp
        original_forward = mlp_module.forward
        mlp_module.forward = wrap_neuron_intervention_forward(original_forward, neuron_list, mode)
        hooks[layer_idx] = original_forward
    
    return hooks


def remove_neuron_intervention_hooks(model, hooks):
    """
    Restores the original forward functions for the MLP modules after neuron intervention.

    Args:
        model: Transformer model.
        hooks (dict): Dictionary mapping layer_idx to the original forward function.
    """
    for layer_idx, original_forward in hooks.items():
        model.model.layers[layer_idx].mlp.forward = original_forward




###########Analyze results#################
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
        # get segment‐wise predictions at the given mode and K
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