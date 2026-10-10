"""
Step 2: Train a reward model on the preference data.

The reward model learns to assign higher scores to 'chosen' completions
and lower scores to 'rejected' completions using a Bradley-Terry loss.
"""
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from config import (
    BASE_MODEL, DATA_DIR, REWARD_MODEL_DIR, METRICS_DIR,
    REWARD_EPOCHS, REWARD_BATCH_SIZE, REWARD_LR, MAX_LENGTH, SEED
)

torch.manual_seed(SEED)


class RewardModel(nn.Module):
    """A reward model that outputs a scalar score for an input sequence."""

    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last non-padding token's hidden state
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = outputs.last_hidden_state[batch_idx, seq_lengths]
        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_enc = self.tokenizer(
            item["chosen"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            item["rejected"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        return {
            "chosen_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_mask": rejected_enc["attention_mask"].squeeze(0),
        }


def compute_loss(reward_chosen, reward_rejected):
    """Bradley-Terry pairwise loss: -log(sigmoid(r_chosen - r_rejected))."""
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected) + 1e-8).mean()


def evaluate(model, dataloader, device):
    model.eval()
    all_correct = 0
    all_total = 0
    total_loss = 0.0
    total_margin = 0.0

    with torch.no_grad():
        for batch in dataloader:
            r_chosen = model(
                batch["chosen_ids"].to(device),
                batch["chosen_mask"].to(device)
            )
            r_rejected = model(
                batch["rejected_ids"].to(device),
                batch["rejected_mask"].to(device)
            )
            loss = compute_loss(r_chosen, r_rejected)
            total_loss += loss.item() * r_chosen.size(0)
            margin = (r_chosen - r_rejected).mean().item()
            total_margin += margin * r_chosen.size(0)
            all_correct += (r_chosen > r_rejected).sum().item()
            all_total += r_chosen.size(0)

    return {
        "loss": total_loss / all_total,
        "accuracy": all_correct / all_total,
        "avg_margin": total_margin / all_total,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    with open(os.path.join(DATA_DIR, "preference_train.json"), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(DATA_DIR, "preference_val.json"), "r") as f:
        val_data = json.load(f)

    train_ds = PreferenceDataset(train_data, tokenizer, MAX_LENGTH)
    val_ds = PreferenceDataset(val_data, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=REWARD_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=REWARD_BATCH_SIZE)

    # Init model
    model = RewardModel(BASE_MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=REWARD_LR)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_margin": []}

    print(f"\nTraining reward model for {REWARD_EPOCHS} epochs...")
    for epoch in range(REWARD_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{REWARD_EPOCHS}")
        for batch in pbar:
            r_chosen = model(
                batch["chosen_ids"].to(device),
                batch["chosen_mask"].to(device)
            )
            r_rejected = model(
                batch["rejected_ids"].to(device),
                batch["rejected_mask"].to(device)
            )
            loss = compute_loss(r_chosen, r_rejected)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / n_batches
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_margin"].append(val_metrics["avg_margin"])

        print(f"  Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val Margin: {val_metrics['avg_margin']:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(REWARD_MODEL_DIR, "reward_model.pt"))
    tokenizer.save_pretrained(REWARD_MODEL_DIR)
    print(f"\nReward model saved to {REWARD_MODEL_DIR}")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_accuracy"], "g-o")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    axes[2].plot(history["val_margin"], "r-o")
    axes[2].set_title("Avg Reward Margin (chosen - rejected)")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(os.path.join(METRICS_DIR, "reward_model_training.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {METRICS_DIR}/reward_model_training.png")

    # Save metrics
    with open(os.path.join(METRICS_DIR, "reward_model_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
