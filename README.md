# Traffic-Diff
## Overview

Traffic-Diff is a graph-conditioned denoising diffusion framework for encrypted traffic classification. It integrates temporal traffic representation learning, wavelet-based feature extraction, Flow Interaction Graph Reasoning (FIGR), and curriculum graph-conditioned denoising into a unified framework.

Unlike conventional traffic classification methods that primarily focus on individual flows, Traffic-Diff explicitly models interactions among flows to capture cross-flow dependencies.

The framework consists of four main components:

1. **Relational Traffic Construction:** Extracts temporal and wavelet-based representations from raw network traffic.
2. **Flow Interaction Graph Reasoning (FIGR):** Models relationships among traffic flows through typed graph attention and message passing.
3. **Curriculum Graph-Conditioned Denoising:** Progressively introduces packet masking, scalogram block masking, and graph relation masking to learn robust representations.
4. **Supervised Fine-tuning:** Uses the learned graph-aware representations for encrypted traffic classification.

## Framework Architecture

The overall framework is illustrated below.

![Traffic-Diff Architecture](overview.png)

### Main Components

- **Temporal Encoder:** Extracts packet-level temporal representations from raw traffic bytes.
- **Contextual Encoder:** Captures multi-scale traffic characteristics from wavelet scalograms.
- **Flow Interaction Graph Reasoning (FIGR):** Encodes cross-flow relationships through typed graph construction, temporal relation encoding, and graph message passing.
- **Curriculum Graph-Conditioned Denoising:** Performs dual-domain reconstruction and graph relation recovery under progressively increasing corruption.
- **Classification Head:** Maps graph-aware flow representations to traffic categories.

## Repository Structure

```text
Traffic-Diff/
├── config.py          # Experimental configurations and hyperparameters
├── build_dataset.py   # PCAP processing and dataset construction
├── model.py           # Traffic-Diff model architecture
├── train.py           # Pretraining, fine-tuning, and evaluation
└──  README.md          # Project documentation
Denoise/               # Raw PCAP dataset (user-provided)
Denoise-data/          # Processed dataset (generated)
```

## Requirements

The implementation is based on Python and PyTorch.

Recommended environment:

- Python 3.9+
- PyTorch
- NumPy
- Scapy
- PyWavelets
- scikit-learn

Install the required packages:

```bash
pip install numpy scapy PyWavelets scikit-learn
```

Install a PyTorch version compatible with your CUDA environment by following the instructions at:

https://pytorch.org/get-started/locally/

## Dataset Preparation

Traffic-Diff uses PCAP files as input.

1. Download and prepare the traffic dataset.
2. Place the PCAP files in the directory specified by `DENOISE_DIR` in `config.py`.
3. Ensure that the PCAP filenames contain the corresponding traffic class labels, as required by the dataset construction script.
4. Configure the dataset and output paths in `config.py`.

For example:

```python
DENOISE_DIR = 'Denoise'
DATA_DIR = 'Denoise-data'
```

The dataset construction script extracts:

- Raw packet-byte representations.
- Wavelet-based traffic representations.
- Traffic class labels.
- Flow metadata, including source/destination identifiers and temporal information.

The processed dataset is saved as:

```text
Denoise-data/
├── full_data.npz
└── classes.txt
```

Run the following command to construct the dataset:

```bash
python build_dataset.py
```

## Dataset Download
Download the DenoiseNet dataset from the following link:
- [Dataset (Google Drive)](https://drive.google.com/drive/folders/1-WyNwdKBkydfmPJ3Mv0G-jRZ6TtxyT3O?usp=sharing)

## Training and Evaluation

After generating the processed dataset, run:

```bash
python train.py
```

The training pipeline includes two stages:

1. **Self-supervised Pretraining:** Learns robust traffic representations through curriculum graph-conditioned denoising and reconstruction objectives.
2. **Supervised Fine-tuning:** Fine-tunes the learned representations for encrypted traffic classification.

### Important Hyperparameters

The main experimental settings can be adjusted in `config.py`.

## Experimental Outputs

The training script saves model checkpoints and evaluation results in the configured output directory.

Typical outputs include:

- Pretrained model checkpoint.
- Best fine-tuned model checkpoint.
- Training logs.
- Classification evaluation metrics.
