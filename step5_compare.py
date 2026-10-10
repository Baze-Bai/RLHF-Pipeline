"""
Step 5: Three-way comparison — Base / PPO-v1 (no SFT) / SFT+PPO-v2.

Generates:
  - outputs/metrics/comparison.png   (reward bars + quality panel)
  - outputs/metrics/comparison.json  (all numbers for the report)
"""
import json, os, re
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from config import (
    BASE_MODEL, REWARD_MODEL_DIR, PPO_MODEL_DIR, SFT_MODEL_DIR,
    PPO_V2_MODEL_DIR, METRICS_DIR, MAX_LENGTH, SEED,
)

torch.manual_seed(SEED)

PROMPTS = [
    "def add(a, b):\n    ",
    "def multiply(x, y):\n    ",
    "def is_even(n):\n    ",
    "def reverse_string(s):\n    ",
    "def find_max(lst):\n    ",
    "def factorial(n):\n    ",
    "def sort_list(data):\n    ",
    "def count_words(text):\n    ",
    "def flatten(nested_list):\n    ",
    "def read_file(path):\n    ",
]

GEN_KW = dict(max_new_tokens=80, do_sample=False,
              pad_token_id=None)   # filled per-model below


# ── Models ────────────────────────────────────────────────────────────────────
class RewardModel(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(base)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        slen   = attention_mask.sum(1) - 1
        bidx   = torch.arange(input_ids.size(0), device=input_ids.device)
        hidden = out.last_hidden_state[bidx, slen]
        return self.reward_head(hidden).squeeze(-1)


def load_rm(device):
    rm = RewardModel(BASE_MODEL).to(device)
    rm.load_state_dict(torch.load(
        os.path.join(REWARD_MODEL_DIR, "reward_model.pt"),
        map_location=device, weights_only=True))
    rm.eval()
    for p in rm.parameters(): p.requires_grad_(False)
    return rm


def score_texts(rm, tok, texts, device):
    enc = tok(texts, truncation=True, max_length=MAX_LENGTH,
               padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        return rm(enc["input_ids"], enc["attention_mask"])


# ── Heuristic quality ─────────────────────────────────────────────────────────
def heuristic_quality(text: str, prompt: str) -> float:
    """Score 0-8 based on structural docstring features."""
    doc = text[len(prompt):] if text.startswith(prompt) else text
    score = 0.0
    if '"""' in doc or "'''" in doc:       score += 2.0
    if "Args:" in doc:                      score += 1.5
    if "Returns:" in doc:                   score += 1.5
    words = doc.split()
    if len(words) >= 10:                    score += 1.0
    nls = doc.count("\n")
    if 2 <= nls <= 15:                      score += 1.0
    non_ws = len(doc.replace(" ", "").replace("\n", "").replace("\t", ""))
    if non_ws > 20:                         score += 0.5
    return min(score, 8.0)


# ── Generate texts from a model dir ───────────────────────────────────────────
def generate_all(model_dir: str, device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    lm.eval()
    texts = []
    kw = {**GEN_KW, "pad_token_id": tok.eos_token_id}
    with torch.no_grad():
        for pr in PROMPTS:
            ids = tok.encode(pr, return_tensors="pt").to(device)
            out = lm.generate(ids, **kw)
            texts.append(tok.decode(out[0], skip_special_tokens=True))
    return texts, tok


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rm = load_rm(device)

    # Load a common tokeniser for scoring (base model vocab)
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token

    models = {
        "Base\n(no SFT,\nno PPO)":  BASE_MODEL,
        "PPO-v1\n(no SFT)":         PPO_MODEL_DIR,
        "SFT +\nPPO-v2":            PPO_V2_MODEL_DIR,
    }

    results = {}
    for label, model_dir in models.items():
        tag = label.replace("\n", " ").replace("(", "").replace(")", "").strip()
        print(f"\n── {tag} ──")
        texts, tok = generate_all(model_dir, device)
        scores = score_texts(rm, base_tok, texts, device)
        quals  = [heuristic_quality(t, PROMPTS[i]) for i, t in enumerate(texts)]

        mean_r = scores.mean().item()
        std_r  = scores.std().item()
        mean_q = float(np.mean(quals))
        std_q  = float(np.std(quals))

        print(f"  reward:  {mean_r:.4f} ± {std_r:.4f}")
        print(f"  quality: {mean_q:.2f} ± {std_q:.2f}")

        for i, (pr, txt, sc, qu) in enumerate(zip(PROMPTS, texts, scores, quals)):
            print(f"  [{i}] reward={sc:.3f} quality={qu:.1f}")
            print(f"       {txt[:120].replace(chr(10), ' ')}")

        results[tag] = {
            "mean_reward": mean_r, "std_reward": std_r,
            "mean_quality": mean_q, "std_quality": std_q,
            "examples": [
                {"prompt": PROMPTS[i].strip(), "text": texts[i],
                 "reward": scores[i].item(), "quality": quals[i]}
                for i in range(len(PROMPTS))
            ]
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    labels = list(results.keys())
    means_r = [results[l]["mean_reward"]  for l in labels]
    stds_r  = [results[l]["std_reward"]   for l in labels]
    means_q = [results[l]["mean_quality"] for l in labels]
    stds_q  = [results[l]["std_quality"]  for l in labels]

    # Short x-axis labels
    xlabels = ["Base\n(no SFT,\nno PPO)", "PPO-v1\n(no SFT)", "SFT +\nPPO-v2"]
    colors  = ["#aaaaaa", "#ff9f4f", "#7fbf7f"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars0 = axes[0].bar(xlabels, means_r, yerr=stds_r,
                        color=colors, capsize=10, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Mean Reward Model Score\n(higher = better)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Reward Score")
    for bar, val in zip(bars0, means_r):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylim(bottom=min(0, min(means_r) - 0.5))

    bars1 = axes[1].bar(xlabels, means_q, yerr=stds_q,
                        color=colors, capsize=10, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Mean Heuristic Quality Score\n(higher = better, max=8)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Quality Score (0-8)")
    axes[1].set_ylim(0, 9)
    for bar, val in zip(bars1, means_q):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.suptitle("Three-Way Model Comparison: Base vs PPO-v1 vs SFT+PPO-v2",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(METRICS_DIR, "comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison chart saved → {out_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(METRICS_DIR, "comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Comparison data saved → {json_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label in labels:
        r = results[label]
        print(f"  {label:30s}  reward={r['mean_reward']:+.4f}  quality={r['mean_quality']:.2f}")


if __name__ == "__main__":
    main()
