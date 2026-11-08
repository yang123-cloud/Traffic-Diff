"""Configuration for FIGR-enhanced Traffic-Diff."""
import os

class Config:
    # Paths are relative to the working directory (run from the project folder).
    DENOISE_DIR = "Denoise"       #Denoise   #Edge-IIoTset #CIC_IOT #CICAPT
    DATA_DIR = "Denoise-data"
    DATA_FILE = "full_data.npz"

    # Flow extraction
    MAX_PACKETS = 64
    MIN_PACKETS = 5
    PACKET_BYTES = 64
    MAX_FLOWS_PER_CLASS = 500
    CONTEXT_WINDOW = 1.0
    MAX_PACKETS_PER_PCAP = 2_000_000

    # Wavelet
    WAVELET = "cgau1"
    SCALES = 64

    # Model
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 3
    DROPOUT = 0.10

    # Curriculum packet / wavelet masking
    MASK_RATIO = 0.75
    MASK_RATIO_START = 0.15
    MASK_RATIO_END = 0.75
    GRAPH_MASK_START = 0.05
    GRAPH_MASK_END = 0.30
    GRAPH_K_NEIGHBORS = 2
    GRAPH_RELATION_LOSS_WEIGHT = 0.20
    WAVELET_LOSS_WEIGHT = 0.50
    AUX_LOSS_WEIGHT = 0.05

    # Training
    BATCH_SIZE = 32
    PRETRAIN_EPOCHS = 150
    FINETUNE_EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15
    EARLY_STOP_ACC = 0.97  # training ACC only; test remains held out until final evaluation
    DEVICE = "cuda"
    SEED = 42
    NUM_WORKERS = 0

    # Required split: 70% train / 30% test
    TEST_SIZE = 0.30

cfg = Config()
os.makedirs(cfg.DATA_DIR, exist_ok=True)


