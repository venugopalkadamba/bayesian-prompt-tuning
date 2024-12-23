# Exploring Embedding Priors in Prompt-Tuning for Improved Interpretability and Control

This repository contains the code, experiments, and results for our research project: **Exploring Embedding Priors in Prompt-Tuning for Improved Interpretability and Control**. This work investigates the use of Bayesian priors to address embedding collapse in prompt-tuning and enhance the adaptability, interpretability, and generalization capabilities of large language models.

## Overview
Prompt-tuning is a parameter-efficient method for adapting pre-trained language models to new tasks. However, embedding collapse—a phenomenon where embeddings converge into specific clusters—limits generalization and flexibility. Our research explores embedding priors to mitigate this issue and examines their impact on downstream tasks.

## Key Features
- **Bayesian Priors**: Implementations of Gaussian, structured, exclusion, and interpolation priors.
- **Prompt-Tuning Techniques**: Experiments with Soft Prompt-Tuning and Deep Prompt-Tuning.
- **Datasets**: Focused on **SQuAD (Stanford Question Answering Dataset)** and **DeepMind MATH**.
- **Visualizations**: t-SNE and PCA plots for analyzing activation clusters and embedding distributions.
- **Applications**: Insights into Chains-of-Thought distillation and multimodal task generalization.

## Repository Structure
```plaintext
├── code
│   ├── prompt_tuning.py      # Core prompt-tuning implementation
│   ├── prior_design.py       # Embedding prior design and sampling
│   ├── experiments.py        # Experiment setup and evaluation
│   ├── visualization.py      # Scripts for t-SNE and PCA plots
│   └── utils.py              # Helper functions and utilities
├── data
│   ├── squad/                # Stanford Question Answering Dataset
│   ├── math/                 # DeepMind MATH Dataset
│   └── preprocessed/         # Preprocessed embeddings and activations
├── results
│   ├── plots/                # Visualizations (t-SNE, PCA, etc.)
│   └── logs/                 # Experiment logs and metrics
├── README.md                 # Project description and usage guide
├── requirements.txt          # Python dependencies
└── LICENSE                   # License for the repository
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/prompt-tuning-priors.git
   cd prompt-tuning-priors
   ```
2. Set up a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running Experiments
1. Preprocess datasets:
   ```bash
   python code/preprocess_data.py
   ```
2. Run prompt-tuning experiments:
   ```bash
   python code/experiments.py --task squad --prior gaussian --tuning soft
   ```
3. Generate visualizations:
   ```bash
   python code/visualization.py --input results/logs/squad_experiment.json
   ```

### Key Parameters
- `--task`: Specify the task (e.g., `squad`, `math`).
- `--prior`: Select the embedding prior (e.g., `gaussian`, `structured`).
- `--tuning`: Choose the tuning method (`soft`, `deep`).

## Results
Key findings:
- Priors significantly influence embedding distributions and can mitigate collapse.
- Models adapt effectively to embeddings sampled from diverse regions of activation space.
- Distinct activation clusters are observed for different tasks (e.g., NLP vs. arithmetic).

## Future Work
- Extending priors for multimodal tasks.
- Exploring Chains-of-Thought distillation applications.
- Investigating activation-level regularizations for improved domain integration.

## Contributors
- **Sergey Sedov** ([ss19021@nyu.edu](mailto:ss19021@nyu.edu))
- **Venu Gopal Kadamba** ([vk2636@nyu.edu](mailto:vk2636@nyu.edu))
- **Sumanth Bharadwaj Hachalli Karanam** ([sh8111@nyu.edu](mailto:sh8111@nyu.edu))


---
Feel free to contribute to this project by opening issues or submitting pull requests. For any inquiries or collaboration opportunities, reach out via email or LinkedIn!
