"""
Centralized configuration for the RLHF pipeline.
V2 adds SFT stage and improved PPO settings.
"""
import os

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
REWARD_MODEL_DIR = os.path.join(OUTPUT_DIR, "reward_model")
PPO_MODEL_DIR = os.path.join(OUTPUT_DIR, "ppo_model")
SFT_MODEL_DIR = os.path.join(OUTPUT_DIR, "sft_model")
PPO_V2_MODEL_DIR = os.path.join(OUTPUT_DIR, "ppo_v2_model")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# Model
BASE_MODEL = "distilgpt2"  # Toy LLM for fast iteration

# Data
NUM_PREFERENCE_PAIRS = 2000
MAX_LENGTH = 128
SEED = 42

# Reward model training
REWARD_EPOCHS = 3
REWARD_BATCH_SIZE = 16
REWARD_LR = 1e-5

# PPO training (v1 — baseline, no SFT)
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 8
PPO_MINI_BATCH_SIZE = 4
PPO_LR = 1.41e-5
PPO_STEPS = 200
KL_PENALTY = "kl"
TARGET_KL = 6.0
INIT_KL_COEF = 0.2

# SFT training
SFT_EPOCHS = 5
SFT_BATCH_SIZE = 16
SFT_LR = 5e-5
SFT_MAX_LENGTH = 160   # longer to fit full docstring

# PPO v2 (uses SFT checkpoint as base + KL reference)
PPO_V2_EPOCHS = 4
PPO_V2_BATCH_SIZE = 8
PPO_V2_MINI_BATCH_SIZE = 4
PPO_V2_LR = 1.41e-5
PPO_V2_STEPS = 400          # 2× more steps
PPO_V2_TARGET_KL = 6.0
PPO_V2_INIT_KL_COEF = 0.1
PPO_V2_ENTROPY_COEF = 0.05  # 5× larger — prevents mode collapse
PPO_V2_GEN_MAX_NEW = 80     # longer responses now that model can generate them

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR, REWARD_MODEL_DIR, PPO_MODEL_DIR,
          SFT_MODEL_DIR, PPO_V2_MODEL_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)
