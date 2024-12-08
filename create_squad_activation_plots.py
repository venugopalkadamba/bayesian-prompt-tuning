import argparse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_c4_activations(model, dataset, tokenizer, MAX_C4_SAMPLES, LAYER_IDX):
    print(f"Collecting {MAX_C4_SAMPLES} samples' activations from C4")

    upper_layer_trajectories = []

    device = 'cuda:0'
    model = model.to(device)
    model.eval()

    for i, sample in enumerate(dataset):
        # Tokenize the input text
        inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            # Pass input through the model to get hidden states
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Get the upper layer's embeddings (last layer hidden states)
            upper_layer_embeds = hidden_states[LAYER_IDX].squeeze(0)  # Shape: (seq_len, hidden_dim)

            # Append token indices (trajectory) and corresponding embeddings
            upper_layer_trajectories.append((input_ids.squeeze(0).tolist(), upper_layer_embeds.cpu().numpy()))

        if i % 100 == 0:
            print(f"Processed {i+1} samples out of {len(dataset)}")
        
        if i >= MAX_C4_SAMPLES:
            break

    print(f"Collected {len(upper_layer_trajectories)} upper-layer trajectories.")

    upper_activations = np.concatenate(
        [traj[1] for traj in upper_layer_trajectories],
        axis=0
    )
    token_trajectories = np.concatenate([
        np.array(traj[0])
        for traj in upper_layer_trajectories
    ], axis=0)

    return upper_activations, token_trajectories


def preprocess_function(examples, model, tokenizer):
    prompt_template = "{context}\nQuestion: {question}\nAnswer:"
    inputs = [
        prompt_template.format(context=c, question=q)
        for c, q in zip(examples["context"], examples["question"])
    ]

    answers = [answer['text'][0] for answer in examples["answers"]]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(answers, max_length=128, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]

    for key in model_inputs.keys():
        model_inputs[key] = model_inputs[key].to(model.device)
    labels = labels.to(model.device)

    model_inputs["labels"] = labels

    model_inputs["labels"] = F.pad(
        model_inputs["labels"],
        (0, model_inputs["input_ids"].size(1) - model_inputs["labels"].size(1)),
        value=-100
    )

    return model_inputs
    
    
def collect_squad_activations(model, squad_dataset, tokenizer, MAX_SQUAD_SAMPLES, LAYER_IDX):
    class SquadCollateClass():
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
        
        def __call__(self, dataset_items):
            batch = {
                "context": [sample["context"] for sample in dataset_items],
                "question": [sample["question"] for sample in dataset_items],
                "answers": [sample["answers"] for sample in dataset_items],
            }
            model_inputs = preprocess_function(batch, self.model, self.tokenizer)
            return model_inputs

    device = 'cuda:0'
    model = model.to(device)

    def collect_upper_layer_trajectories(model, dataloader, extend_with_labels=False, max_samples=1000, layer_idx=-1):
        upper_layer_trajectories = []
        upper_layer_trajectories_with_labels = []
        model.eval()

        for i, model_inputs in enumerate(dataloader):
            for k in model_inputs.keys():
                model_inputs[k] = model_inputs[k].to(device)
            
            labels_clean = model_inputs['labels'].clone()
            labels_clean = labels_clean[:, :128]
            if (labels_clean == -100).sum():
                labels_clean[labels_clean == -100] = tokenizer.pad_token_id
                print('replaced -100 with pad_token_id')

            input_ids_list = [model_inputs["input_ids"]]
            trajectories_list = [upper_layer_trajectories]
            if extend_with_labels:
                extended_input_ids = torch.cat([model_inputs["input_ids"], labels_clean], dim=1)
                input_ids_list.append(extended_input_ids)
                trajectories_list.append(upper_layer_trajectories_with_labels)

            with torch.no_grad():
                for input_ids, trajectories in zip(input_ids_list, trajectories_list):
                    outputs = model(input_ids=input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states

                    last_layer_embeds = hidden_states[layer_idx]

                    for idx in range(input_ids.size(0)):
                        tokens_input_ids = np.array([
                            x.detach().cpu().numpy()
                            for x in input_ids[idx]
                        ])
                        embeddings = last_layer_embeds[idx].detach().cpu().numpy()
                        trajectories.append((tokens_input_ids, embeddings))

            # Optional progress report
            if i % 50 == 0:
                processed = (i + 1) * input_ids.size(0)
                print(f"Processed {processed} samples out of {min(len(squad_dataset), max_samples)}")

            if (i + 1) * dataloader.batch_size >= max_samples:
                break

        print(f"Collected {len(upper_layer_trajectories)} upper-layer trajectories.")
        if extend_with_labels:
            return upper_layer_trajectories, upper_layer_trajectories_with_labels
        return upper_layer_trajectories

    squad_collate = SquadCollateClass(model, tokenizer)
    dataloader = DataLoader(squad_dataset, batch_size=2, shuffle=True, collate_fn=squad_collate)
    upper_layer_trajectories_squad, upper_layer_trajectories_with_labels_squad = collect_upper_layer_trajectories(
        model, dataloader, extend_with_labels=True, max_samples=MAX_SQUAD_SAMPLES, layer_idx=LAYER_IDX
    )

    token_trajectories_squad = np.array([
        np.array(traj[0])
        for traj in upper_layer_trajectories_squad
        ])
    activations_squad = np.concatenate([traj[1][None, ...] for traj in upper_layer_trajectories_squad], axis=0)

    token_trajectories_with_labels_squad = np.array([
        np.array(traj[0])
        for traj in upper_layer_trajectories_with_labels_squad
    ])
    activations_with_labels_squad = np.concatenate([traj[1][None, ...] for traj in upper_layer_trajectories_with_labels_squad], axis=0)

    return token_trajectories_squad, activations_squad, token_trajectories_with_labels_squad, activations_with_labels_squad


def main(
        HF_TOKEN,
        model_name="meta-llama/Llama-3.2-1B",
        MAX_C4_SAMPLES=250,
        MAX_SQUAD_SAMPLES=250,
        LAYER_IDX=-1,
        max_generative_steps=20,
        save_activations=False,
        load_activations=False,
        load_paths={}
):
    if load_activations and len(load_paths) == 0:
        load_paths = {
            "token_trajectories_c4": "tokens_on_input_ids_c4.npy",
            "activations_c4": f"activations_{LAYER_IDX}_on_input_ids_c4.npy",
            "token_trajectories_squad": "tokens_on_input_ids_squad.npy",
            "token_trajectories_with_labels_squad": "tokens_on_input_ids_labels_squad.npy",
            "activations_squad": f"activations_{LAYER_IDX}_on_input_ids_squad.npy",
            "activations_with_labels_squad": f"activations_{LAYER_IDX}_on_input_ids_labels_squad.npy"    
        }

    model = AutoModelForCausalLM.from_pretrained(model_name,token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=HF_TOKEN)
    N_ACTIVATION_LAYERS = model.config.num_hidden_layers + 1

    vocab = list(tokenizer.get_vocab().keys())

    path = "allenai/c4"
    name = None
    streaming = False
    split = "train"
    data_files = {"train": "en/c4-train.00000-of-01024.json.gz"}
    max_samples = 1000

    shuffle_seed = 52

    dataset = load_dataset(path, name=name, split=split, data_files=data_files)
    dataset = dataset.shuffle(shuffle_seed)
    dataset = dataset.select(range(max_samples))

    squad_dataset = load_dataset("squad", split='train')
    squad_dataset = squad_dataset.shuffle(shuffle_seed)
    squad_dataset = squad_dataset.select(range(max_samples))

    tokenizer = AutoTokenizer.from_pretrained(model_name,token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    if load_activations:
        upper_activations = np.load(load_paths["activations_c4"])
        token_trajectories = np.load(load_paths["token_trajectories_c4"])
        token_trajectories_squad = np.load(load_paths["token_trajectories_squad"])
        token_trajectories_with_labels_squad = np.load(load_paths["token_trajectories_with_labels_squad"])
        activations_squad = np.load(load_paths["activations_squad"])
        activations_with_labels_squad = np.load(load_paths["activations_with_labels_squad"])
    else:
        upper_activations, token_trajectories = collect_c4_activations(
            model, dataset, tokenizer, MAX_C4_SAMPLES, LAYER_IDX
        )
        output = collect_squad_activations(
            model, squad_dataset, tokenizer, MAX_SQUAD_SAMPLES, LAYER_IDX
        )
        token_trajectories_squad, activations_squad, token_trajectories_with_labels_squad, activations_with_labels_squad = output

    if (not load_activations) and save_activations:
        np.save(f'tokens_on_input_ids_c4.npy', token_trajectories)
        np.save(f'activations_{LAYER_IDX}_on_input_ids_c4.npy', upper_activations)
        np.save(f'tokens_on_input_ids_squad.npy', token_trajectories_squad)
        np.save(f'tokens_on_input_ids_labels_squad.npy', token_trajectories_with_labels_squad)
        np.save(f'activations_{LAYER_IDX}_on_input_ids_squad.npy', activations_squad)
        np.save(f'activations_{LAYER_IDX}_on_input_ids_labels_squad.npy', activations_with_labels_squad)
    
    B_orig, D = upper_activations.shape
    B_squad, L_squad, D = activations_squad.shape
    B_squad_labels, L_squad_labels, D = activations_with_labels_squad.shape

    # concatenated with ones from clean input_ids generation
    plot_activations = np.concatenate([
        upper_activations,
        activations_squad.reshape(B_squad * L_squad, D),
        activations_with_labels_squad.reshape(B_squad_labels * L_squad_labels, D)
    ], axis=0)

    pca = PCA(n_components=2)
    pca_coords_ = pca.fit_transform(plot_activations)  # shape: (total_num_points, 2)

    kicked_indices = np.where(pca_coords_[:, 0] > 100)[0]
    if len(kicked_indices):
        pca_coords_filtered = pca_coords_.copy()
        pca_coords_filtered[kicked_indices] = pca_coords_filtered[~kicked_indices].mean(axis=0)
        filter_names = ["unfiltered", "filtered"]
        pca_coords_list = [pca_coords_, pca_coords_filtered]
    else:
        filter_names = ["unfiltered"]
        pca_coords_list = [pca_coords_]
    
    for filter_name, pca_coords in zip(filter_names, pca_coords_list):
        orig_coords = pca_coords[:B_orig]
        squad_coords = pca_coords[B_orig:B_orig + B_squad * L_squad]
        squad_labels_coords = pca_coords[B_orig + B_squad * L_squad:]

        plt.figure(figsize=(10,10))

        plt.scatter(orig_coords[:,0], orig_coords[:,1], c='gray', alpha=0.5, label='C4 activations of input_ids')
        plt.scatter(squad_labels_coords[:,0], squad_labels_coords[:,1], c='mediumpurple', alpha=0.5, label='SQUAD activations of input_ids + labels')
        plt.scatter(squad_coords[:,0], squad_coords[:,1], c='cornflowerblue', alpha=0.5, label='SQUAD activations of input_ids')

        plt.legend(loc="upper right", bbox_to_anchor=(1.5,1.0))
        plt.title(f"LLaMa-3.2 1B {LAYER_IDX} / {N_ACTIVATION_LAYERS - 1} layer's activations")
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.grid(True)
        plt.savefig(f"LLAMA_{LAYER_IDX}_layer_activations_C4_SQUAD_{filter_name}.png")
        plt.close()

    
    # WITH SAMPLE TRAJECTORY

    device = 'cuda:0'
    model = model.to(device)
    model.eval()

    random_sample_index = np.random.randint(len(squad_dataset))
    sample = squad_dataset[random_sample_index]

    # Tokenize the input text
    model_inputs = preprocess_function({
        "context": [sample["context"]],
        "question": [sample["question"]],
        "answers": [sample["answers"]]
    }, model, tokenizer)

    input_ids = model_inputs["input_ids"].to(device)
    labels = model_inputs["labels"].to(device)

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of layers
        last_layer_embeds_clean = hidden_states[LAYER_IDX].squeeze(0).cpu().numpy()

    all_embeddings = []
    all_embeddings.extend(last_layer_embeds_clean)

    max_steps = min(labels.shape[1], max_generative_steps)
    trajectories = [[] for _ in range(max_steps)]

    for gen_step in range(1, max_steps + 1):
        extended_input = torch.cat([input_ids, labels[:,:gen_step]], dim=1)

        with torch.no_grad():
            outputs = model(extended_input, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_layer_embeds = hidden_states[LAYER_IDX].squeeze(0).cpu().numpy()
        
        seq_len_context = input_ids.size(1)
        for lbl_idx in range(gen_step):
            label_pos = seq_len_context + lbl_idx
            label_embedding = last_layer_embeds[label_pos]
            trajectories[lbl_idx].append(label_embedding)
            all_embeddings.append(label_embedding)

    all_embeddings = np.array(all_embeddings)

    B_orig, D = upper_activations.shape
    B_squad, L_squad, D = activations_squad.shape
    B_squad_labels, L_squad_labels, D = activations_with_labels_squad.shape

    ALL_activations = np.concatenate([
        upper_activations,
        activations_squad.reshape(B_squad * L_squad, D),
        activations_with_labels_squad.reshape(B_squad_labels * L_squad_labels, D),
        all_embeddings
    ], axis=0)

    pca = PCA(n_components=2)
    pca_coords_ = pca.fit_transform(ALL_activations)

    kicked_indices = np.where(pca_coords_[:, 0] > 100)[0]
    if len(kicked_indices):
        pca_coords_filtered = pca_coords_.copy()
        pca_coords_filtered[kicked_indices] = pca_coords_filtered[~kicked_indices].mean(axis=0)
        filter_names = ["unfiltered", "filtered"]
        pca_coords_list = [pca_coords_, pca_coords_filtered]
    else:
        filter_names = ["unfiltered"]
        pca_coords_list = [pca_coords_]
    
    for filter_name, pca_coords in zip(filter_names, pca_coords_list):
        num_clean_orig = B_orig
        num_clean_squad = B_squad * L_squad
        num_with_labels_squad = B_squad_labels * L_squad_labels
        num_baseline = last_layer_embeds_clean.shape[0]

        clean_orig_coords = pca_coords[:num_clean_orig]
        clean_squad_coords = pca_coords[num_clean_orig : num_clean_orig + num_clean_squad]
        with_labels_squad_coords = pca_coords[
            num_clean_orig + num_clean_squad : 
            num_clean_orig + num_clean_squad + num_with_labels_squad
        ]
        baseline_coords = pca_coords[
            num_clean_orig + num_clean_squad + num_with_labels_squad :
            num_clean_orig + num_clean_squad + num_with_labels_squad + num_baseline
        ]

        label_coords = pca_coords[num_clean_orig + num_clean_squad + num_with_labels_squad + num_baseline:]

        label_coords_trajectories = []
        for lbl_idx in range(max_steps):
            label_coords_trajectories.append(label_coords[-max_steps+lbl_idx].reshape(1, -1))

        final_label_coords = label_coords[-max_steps:]

        plt.figure(figsize=(10,10))

        plt.scatter(clean_orig_coords[:,0], clean_orig_coords[:,1], c='gray', alpha=0.5, label='C4 activations of input_ids')
        plt.scatter(with_labels_squad_coords[:,0], with_labels_squad_coords[:,1], c='mediumpurple', alpha=0.5, label='SQUAD activations of input_ids + labels')
        plt.scatter(clean_squad_coords[:,0], clean_squad_coords[:,1], c='cornflowerblue', alpha=0.5, label='SQUAD activations of input_ids')
        plt.scatter(baseline_coords[:,0], baseline_coords[:,1], c='darkslategray', alpha=0.5, label='Sample activations of input_ids')

        colors = plt.cm.viridis(np.linspace(0.7, 1, max_steps))
        start_idx = 0

        for step_i in range(0, max_generative_steps):
            start_idx = step_i
            end_idx = start_idx + 1
            coords_for_this_step = final_label_coords[start_idx:end_idx]

            plt.scatter(coords_for_this_step[:,0], coords_for_this_step[:,1],
                        color=colors[step_i], marker='o', label=f'Sample activations of label[{lbl_idx}]')

            if step_i == 0:
                continue

            start_x, start_y = final_label_coords[step_i-1]
            end_x, end_y = final_label_coords[step_i]
            plt.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                            arrowprops=dict(arrowstyle="->", color=colors[step_i]))

        plt.legend(loc="upper right", bbox_to_anchor=(1.5,1.0))
        plt.title(f"LLaMa-3.2 1B {LAYER_IDX} / {N_ACTIVATION_LAYERS - 1} layer's activations with sample labels trajectory")
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.grid(True)
        plt.savefig(f"LLAMA_{LAYER_IDX}_layer_activations_with_labels_trajectory_C4_SQUAD_{filter_name}.png")
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLaMA activation analysis.")
    parser.add_argument("--HF_TOKEN", type=str, required=True, help="Hugging Face token for model loading")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path")
    parser.add_argument("--MAX_C4_SAMPLES", type=int, default=250, help="Max samples from C4")
    parser.add_argument("--MAX_SQUAD_SAMPLES", type=int, default=250, help="Max samples from SQuAD")
    parser.add_argument("--LAYER_IDX", type=int, default=-1, help="Index of the layer to visualize")
    parser.add_argument("--max_generative_steps", type=int, default=20, help="Max generative steps for label trajectories")
    parser.add_argument("--save_activations", action="store_true", help="Whether to save activations to npy files")
    parser.add_argument("--load_activations", action="store_true", help="Whether to load pre-saved activations")
    parser.add_argument("--load_paths", type=dict, default={}, help="Load paths, check start of main() for default example")

    args = parser.parse_args()

    main(
        HF_TOKEN=args.HF_TOKEN,
        model_name=args.model_name,
        MAX_C4_SAMPLES=args.MAX_C4_SAMPLES,
        MAX_SQUAD_SAMPLES=args.MAX_SQUAD_SAMPLES,
        LAYER_IDX=args.LAYER_IDX,
        max_generative_steps=args.max_generative_steps,
        save_activations=args.save_activations,
        load_activations=args.load_activations,
        load_paths=args.load_paths
    )
