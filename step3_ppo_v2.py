"""
Step 3 (v2): PPO alignment using SFT model as policy base AND KL reference.

Key improvements over v1:
  - Policy initialised from SFT checkpoint (has code knowledge)
  - KL reference = frozen copy of SFT model (not untrained base)
  - entropy_coef = 0.05  (5× larger — prevents mode collapse)
  - 400 training steps (2× more budget)
  - max_new_tokens = 80  (model can now generate full docstrings)

Result: PPO fine-tunes a model that already knows the docstring format,
        so the reward signal is informative from step 0.
"""
import json, os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    BASE_MODEL, REWARD_MODEL_DIR, SFT_MODEL_DIR, PPO_V2_MODEL_DIR, METRICS_DIR,
    PPO_V2_BATCH_SIZE, PPO_V2_MINI_BATCH_SIZE, PPO_V2_LR, PPO_V2_STEPS, PPO_V2_EPOCHS,
    PPO_V2_INIT_KL_COEF, PPO_V2_TARGET_KL, PPO_V2_ENTROPY_COEF, PPO_V2_GEN_MAX_NEW,
    MAX_LENGTH, SEED,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

CLIP_EPS    = 0.2
VALUE_COEFF = 0.1
MAX_GRAD    = 1.0


# ── Reward model ──────────────────────────────────────────────────────────────
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


# ── Policy + value head ───────────────────────────────────────────────────────
class PolicyWithValue(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.lm    = AutoModelForCausalLM.from_pretrained(model_dir)
        self.vhead = nn.Linear(self.lm.config.n_embd, 1)

    def forward(self, input_ids, attention_mask=None):
        out    = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        logits = out.logits
        if attention_mask is not None:
            slen   = attention_mask.sum(1) - 1
            bidx   = torch.arange(input_ids.size(0), device=input_ids.device)
            last_h = out.hidden_states[-1][bidx, slen]
        else:
            last_h = out.hidden_states[-1][:, -1, :]
        value = self.vhead(last_h).squeeze(-1)
        return logits, value

    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)


# ── Utilities ─────────────────────────────────────────────────────────────────
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


def score_texts(rm, tok, texts, device):
    enc = tok(texts, truncation=True, max_length=MAX_LENGTH,
               padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        return rm(enc["input_ids"], enc["attention_mask"])


def response_logprob(logits, full_ids, query_len):
    """Sum log-prob over response tokens only (positions query_len..end)."""
    start = max(query_len - 1, 0)
    if start >= logits.size(0):
        return torch.tensor(0.0, device=logits.device)
    resp_logits = logits[start:]
    resp_ids    = full_ids[start + 1:]
    if resp_ids.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    lp = F.log_softmax(resp_logits, dim=-1)
    return lp.gather(1, resp_ids.unsqueeze(1)).squeeze(1).sum()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Reward model (frozen)
    print("Loading reward model …")
    rm = RewardModel(BASE_MODEL).to(device)
    rm.load_state_dict(torch.load(
        os.path.join(REWARD_MODEL_DIR, "reward_model.pt"),
        map_location=device, weights_only=True))
    rm.eval()
    for p in rm.parameters(): p.requires_grad_(False)

    # Policy — initialised from SFT checkpoint
    print("Loading SFT model as policy …")
    policy = PolicyWithValue(SFT_MODEL_DIR).to(device)

    # Reference — frozen copy of SFT (not untrained base)
    print("Creating frozen SFT reference …")
    ref = PolicyWithValue(SFT_MODEL_DIR).to(device)
    ref.load_state_dict(copy.deepcopy(policy.state_dict()))
    ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=PPO_V2_LR)

    gen_kw = dict(max_new_tokens=PPO_V2_GEN_MAX_NEW, do_sample=True,
                  temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)

    # ── Baseline (SFT model before PPO) ───────────────────────────────────────
    print("\nBaseline (SFT) scores …")
    policy.eval()
    base_texts = []
    with torch.no_grad():
        for pr in PROMPTS:
            ids = tok.encode(pr, return_tensors="pt").to(device)
            out = policy.generate(ids, **gen_kw)
            base_texts.append(tok.decode(out[0], skip_special_tokens=True))
    b_scores    = score_texts(rm, tok, base_texts, device)
    baseline_mu = b_scores.mean().item()
    print(f"SFT baseline mean reward: {baseline_mu:.4f}  std: {b_scores.std().item():.4f}")
    for i, (pr, txt, sc) in enumerate(zip(PROMPTS, base_texts, b_scores)):
        print(f"  [{i}] {pr.strip()} → reward={sc:.3f}")
        print(f"       {txt[:120].replace(chr(10), ' ')}")

    base_examples = [{"prompt": PROMPTS[i].strip(), "text": base_texts[i],
                      "reward": b_scores[i].item()} for i in range(len(PROMPTS))]

    # ── History ───────────────────────────────────────────────────────────────
    hist = {k: [] for k in ["step", "mean_reward", "std_reward", "max_reward",
                             "min_reward", "kl_divergence", "policy_loss",
                             "value_loss", "entropy", "mean_response_length"]}
    kl_coef = PPO_V2_INIT_KL_COEF

    # ── PPO loop ──────────────────────────────────────────────────────────────
    print(f"\nPPO v2 training for {PPO_V2_STEPS} steps …")

    for step in tqdm(range(PPO_V2_STEPS), desc="PPO-v2"):
        policy.eval()

        # — Rollout —
        prompts_batch = [PROMPTS[(step * PPO_V2_BATCH_SIZE + i) % len(PROMPTS)]
                         for i in range(PPO_V2_BATCH_SIZE)]
        query_ids_list = []
        resp_ids_list  = []
        full_ids_list  = []
        full_texts     = []

        with torch.no_grad():
            for pr in prompts_batch:
                q   = tok.encode(pr, return_tensors="pt").to(device)
                out = policy.generate(q, **gen_kw)
                gen = out[0]
                q_len = q.size(1)
                r   = gen[q_len:]
                if r.numel() == 0:
                    r = torch.tensor([tok.eos_token_id], device=device)
                query_ids_list.append(q.squeeze(0))
                resp_ids_list.append(r)
                full_ids_list.append(gen)
                full_texts.append(tok.decode(gen, skip_special_tokens=True))

        rewards = score_texts(rm, tok, full_texts, device)   # (B,)

        # Padded full-sequence batch
        max_len  = max(f.size(0) for f in full_ids_list)
        full_pad  = torch.zeros(PPO_V2_BATCH_SIZE, max_len, dtype=torch.long, device=device)
        full_mask = torch.zeros(PPO_V2_BATCH_SIZE, max_len, dtype=torch.long, device=device)
        q_lens    = []
        for i, (fid, qid) in enumerate(zip(full_ids_list, query_ids_list)):
            full_pad[i,  :fid.size(0)] = fid
            full_mask[i, :fid.size(0)] = 1
            q_lens.append(qid.size(0))

        # Old log-probs (response tokens only)
        with torch.no_grad():
            old_logits_b, old_values_b = policy(full_pad, full_mask)
            old_lps = torch.stack([
                response_logprob(old_logits_b[i, :-1], full_pad[i], q_lens[i])
                for i in range(PPO_V2_BATCH_SIZE)])

            ref_logits_b, _ = ref(full_pad, full_mask)
            ref_lps = torch.stack([
                response_logprob(ref_logits_b[i, :-1], full_pad[i], q_lens[i])
                for i in range(PPO_V2_BATCH_SIZE)])

        kl_div = (old_lps - ref_lps).clamp(min=0).mean().item()

        # Shaped rewards & advantages
        shaped  = rewards - kl_coef * (old_lps - ref_lps).clamp(min=0).detach()
        returns = shaped.detach()
        adv     = (shaped - shaped.mean()) / (shaped.std() + 1e-8)
        adv     = adv.detach()

        # — PPO update —
        policy.train()
        total_pl, total_vl, total_ent, n_up = 0., 0., 0., 0

        for _ in range(PPO_V2_EPOCHS):
            idx = torch.randperm(PPO_V2_BATCH_SIZE, device=device)
            for start in range(0, PPO_V2_BATCH_SIZE, PPO_V2_MINI_BATCH_SIZE):
                mb      = idx[start:start + PPO_V2_MINI_BATCH_SIZE]
                mb_full = full_pad[mb];    mb_mask = full_mask[mb]
                mb_adv  = adv[mb];         mb_ret  = returns[mb]
                mb_old  = old_lps[mb].detach()
                mb_ql   = [q_lens[i] for i in mb.tolist()]

                new_logits, new_values = policy(mb_full, mb_mask)

                new_lps = torch.stack([
                    response_logprob(new_logits[j, :-1], mb_full[j], mb_ql[j])
                    for j in range(len(mb))])

                # PPO clip
                ratio  = torch.exp((new_lps - mb_old).clamp(-10, 10))
                s1     = ratio * mb_adv
                s2     = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                pl     = -torch.min(s1, s2).mean()

                # Value loss
                vl = F.mse_loss(new_values, mb_ret)

                # Entropy (at last generated token position)
                last_t       = mb_mask.sum(1) - 1
                bidx_        = torch.arange(mb_full.size(0), device=device)
                last_logits  = new_logits[bidx_, last_t]
                probs        = F.softmax(last_logits, dim=-1)
                entropy      = -(probs * (probs + 1e-8).log()).sum(-1).mean()

                loss = pl + VALUE_COEFF * vl - PPO_V2_ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
                optimizer.step()

                total_pl  += pl.item()
                total_vl  += vl.item()
                total_ent += entropy.item()
                n_up      += 1

        # Adaptive KL
        if kl_div > PPO_V2_TARGET_KL * 1.5:
            kl_coef = min(kl_coef * 1.5, 1.0)
        elif kl_div < PPO_V2_TARGET_KL * 0.5:
            kl_coef = max(kl_coef / 1.5, 0.01)

        # Logging
        if step % 10 == 0 or step == PPO_V2_STEPS - 1:
            mr  = rewards.mean().item()
            sr  = rewards.std().item()
            avg = np.mean([r.size(0) for r in resp_ids_list])
            hist["step"].append(step)
            hist["mean_reward"].append(mr)
            hist["std_reward"].append(sr)
            hist["max_reward"].append(rewards.max().item())
            hist["min_reward"].append(rewards.min().item())
            hist["kl_divergence"].append(kl_div)
            hist["policy_loss"].append(total_pl / max(n_up, 1))
            hist["value_loss"].append(total_vl / max(n_up, 1))
            hist["entropy"].append(total_ent / max(n_up, 1))
            hist["mean_response_length"].append(float(avg))
            if step % 50 == 0:
                print(f"  step={step:3d}  r={mr:.4f}  kl={kl_div:.4f}"
                      f"  kl_c={kl_coef:.3f}  ent={total_ent/max(n_up,1):.4f}")

    # ── Post-PPO scores ────────────────────────────────────────────────────────
    print("\nPost-PPO v2 scores …")
    policy.eval()
    post_texts = []
    with torch.no_grad():
        for pr in PROMPTS:
            ids = tok.encode(pr, return_tensors="pt").to(device)
            out = policy.generate(ids, **gen_kw)
            post_texts.append(tok.decode(out[0], skip_special_tokens=True))
    p_scores = score_texts(rm, tok, post_texts, device)
    print(f"Post-PPO v2 mean reward: {p_scores.mean().item():.4f}  std: {p_scores.std().item():.4f}")

    post_examples = [{"prompt": PROMPTS[i].strip(), "text": post_texts[i],
                      "reward": p_scores[i].item()} for i in range(len(PROMPTS))]

    # Save model
    policy.lm.save_pretrained(PPO_V2_MODEL_DIR)
    tok.save_pretrained(PPO_V2_MODEL_DIR)
    print(f"PPO v2 model saved → {PPO_V2_MODEL_DIR}")

    results = {
        "sft_baseline": {"mean_reward": baseline_mu, "std_reward": b_scores.std().item(),
                         "examples": base_examples},
        "post_ppo_v2":  {"mean_reward": p_scores.mean().item(), "std_reward": p_scores.std().item(),
                         "examples": post_examples},
        "training_history": hist,
    }
    with open(os.path.join(METRICS_DIR, "ppo_v2_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Metrics saved → {METRICS_DIR}/ppo_v2_metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    s = hist["step"]

    mr, sr = hist["mean_reward"], hist["std_reward"]
    axes[0, 0].plot(s, mr, "b-", label="Mean")
    axes[0, 0].fill_between(s, [m - v for m, v in zip(mr, sr)],
                                [m + v for m, v in zip(mr, sr)],
                             alpha=0.2, color="blue")
    axes[0, 0].axhline(baseline_mu, color="r", ls="--", label=f"SFT baseline={baseline_mu:.2f}")
    axes[0, 0].set_title("Reward During PPO v2"); axes[0, 0].legend()

    axes[0, 1].plot(s, hist["kl_divergence"], "r-")
    axes[0, 1].axhline(PPO_V2_TARGET_KL, color="gray", ls="--",
                        label=f"Target={PPO_V2_TARGET_KL}")
    axes[0, 1].set_title("KL Divergence"); axes[0, 1].legend()

    axes[0, 2].plot(s, hist["policy_loss"], "m-")
    axes[0, 2].set_title("Policy Loss")

    axes[1, 0].plot(s, hist["value_loss"], "c-")
    axes[1, 0].set_title("Value Loss")

    axes[1, 1].plot(s, hist["entropy"], "g-")
    axes[1, 1].set_title("Policy Entropy")

    axes[1, 2].bar(["SFT Baseline", "After PPO v2"],
                   [baseline_mu, p_scores.mean().item()],
                   yerr=[b_scores.std().item(), p_scores.std().item()],
                   color=["#ff9f4f", "#7fbf7f"], capsize=10)
    axes[1, 2].set_title("SFT Baseline vs PPO v2"); axes[1, 2].set_ylabel("Mean Reward")

    plt.tight_layout()
    fig.savefig(os.path.join(METRICS_DIR, "ppo_v2_training.png"), dpi=150)
    plt.close()
    print(f"Plots saved → {METRICS_DIR}/ppo_v2_training.png")


if __name__ == "__main__":
    main()
