"""
Central configuration for Qwen-RADDino project.
RAD-DINO vision encoder + Qwen2.5-7B-Instruct language model.

Edit the PATHS section below to match your server setup.
"""

import os

# ============================================================
# PATHS — EDIT THESE TO MATCH YOUR SERVER
# ============================================================
DATA_ROOT = "/Nasbackup/lab_kshitij/mariam/data1_backup/varad_mech/IU-Xray"

IMAGE_DIR = "/Nasbackup/lab_kshitij/mariam/data1_backup/varad_mech/IU-Xray/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized"

REPORTS_CSV = "/Nasbackup/lab_kshitij/mariam/data1_backup/varad_mech/processed_reports/cleaned_indiana_reports.csv"

SPLITS_DIR = "/Nasbackup/lab_kshitij/mariam/data1_backup/varad_mech/NewQwen_RadDino_13Apr/data_splits"

# Generated split files (created by prepare_data.py)
# You can REUSE the same splits from the LLaVA-RADDino experiment!
TRAIN_JSON = "/Nasbackup/lab_kshitij/mariam/data1_backup/varad_mech/NewQwen_RadDino_13Apr/data_splits/train.json"
VAL_JSON = "/Nasbackup/lab_kshitij/mariam/data1_backup/varad_mech/NewQwen_RadDino_13Apr/data_splits/val.json"
TEST_JSON = "/Nasbackup/lab_kshitij/mariam/data1_backup/varad_mech/NewQwen_RadDino_13Apr/data_splits/test.json"

OUTPUT_DIR = "/Nasbackup/lab_kshitij/mariam/data1_backup/varad_mech/NewQwen_RadDino_13Apr/outputs/qwen_raddino"

# ============================================================
# MODEL NAMES (HuggingFace Hub IDs)
# ============================================================
RADDINO_MODEL_NAME = "microsoft/rad-dino"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ============================================================
# ARCHITECTURE CONSTANTS
# ============================================================
# RAD-DINO (DINOv2 ViT-B/14)
RADDINO_HIDDEN_SIZE = 768       # RAD-DINO output dim

# Qwen2.5-7B hidden dim (different from Vicuna's 4096!)
LLM_HIDDEN_SIZE = 3584          # Qwen2.5-7B hidden_size

IMAGE_SIZE = 224                # Input image resolution
PATCH_SIZE = 14                 # ViT patch size
NUM_IMAGE_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256

# ============================================================
# TOKENIZATION
# ============================================================
MAX_REPORT_LENGTH = 256

# Qwen2.5 uses ChatML format:
# <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
SYSTEM_PROMPT = "You are a radiology AI assistant. Generate detailed radiology reports from chest X-ray images."
USER_PROMPT = "Generate a detailed radiology report for this chest X-ray."

# ============================================================
# DATA SPLIT RATIOS (used by prepare_data.py)
# ============================================================
TRAIN_RATIO = 0.
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ============================================================
# STAGE 1 — PROJECTOR ALIGNMENT
# ============================================================
STAGE1_LR = 1e-3
STAGE1_EPOCHS = 5
STAGE1_BATCH_SIZE = 4
STAGE1_WARMUP_RATIO = 0.1
STAGE1_WEIGHT_DECAY = 0.0

# ============================================================
# STAGE 2 — LoRA FINE-TUNING
# ============================================================
STAGE2_LR = 2e-5
STAGE2_EPOCHS = 10
STAGE2_BATCH_SIZE = 2
STAGE2_GRAD_ACCUM_STEPS = 8     # Effective batch = 16
STAGE2_WARMUP_RATIO = 0.05
STAGE2_WEIGHT_DECAY = 0.01
STAGE2_LABEL_SMOOTHING = 0.1
STAGE2_MAX_GRAD_NORM = 1.0
STAGE2_PATIENCE = 3

# LoRA hyperparameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Same targets in Qwen2

# ============================================================
# GENERATION SETTINGS
# ============================================================
GEN_MAX_NEW_TOKENS = 256
GEN_NUM_BEAMS = 4
GEN_REPETITION_PENALTY = 1.2
GEN_LENGTH_PENALTY = 1.0

# ============================================================
# IMAGE NORMALIZATION (DINOv2 / ImageNet stats)
# ============================================================
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# ============================================================
# MISC
# ============================================================
SEED = 42
NUM_WORKERS = 4
DTYPE = "bfloat16"
