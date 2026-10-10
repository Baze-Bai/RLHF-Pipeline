"""
Step 3: Manual PPO-based RLHF alignment.

TRL 0.29+ dropped PPOTrainer, so we implement PPO from scratch:
  - Policy LM generates token sequences
  - Reward model scores complete responses
  - Advantage: REINFORCE with mean baseline (standard RLHF trick)
  - PPO clipped surrogate loss (sequence-level)
  - KL penalty against a frozen reference policy
  - Value head trained jointly via MSE on returns

Stability notes (distilGPT-2 on GPU):
  - Log-probs computed only on response tokens (not padding)
  - Gradient norm clipping at 1.0
  - Advantage normalised per mini-batch
  - KL coefficient adapted toward TARGET_KL

Reference: Schulman et al. (2017), Ziegler et al. (2019)
"""
import json, os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import matplotlib.pyplot as plt
from config import (
    BASE_MODEL, REWARD_MODEL_DIR, PPO_MODEL_DIR, METRICS_DIR,
    PPO_BATCH_SIZE, PPO_MINI_BATCH_SIZE, PPO_LR, PPO_STEPS, PPO_EPOCHS,
    INIT_KL_COEF, TARGET_KL, MAX_LENGTH, SEED,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

CLIP_EPS     = 0.2
VALUE_COEFF  = 0.1
ENTROPY_COEF = 0.01
MAX_GRAD     = 1.0
GEN_MAX_NEW  = 40    # shorter → faster + less chance of degenerate sequences


# ── Reward model ──────────────────────────────────────────────────────────
class RewardModel(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.backbone  = AutoModel.from_pretrained(base)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        slen   = attention_mask.sum(1) - 1
        bidx   = torch.arange(input_ids.size(0), device=input_ids.device)
        hidden = out.last_hidden_state[bidx, slen]
        return self.reward_head(hidden).squeeze(-1)


# ── Policy + value head ───────────────────────────────────────────────────
class PolicyWithValue(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.lm    = AutoModelForCausalLM.from_pretrained(base)
        self.vhead = nn.Linear(self.lm.config.n_embd, 1)

    def forward(self, input_ids, attention_mask=None):
        out    = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        logits = out.logits
        # value from the last non-padding token's hidden state
        if attention_mask is not None:
            slen  = attention_mask.sum(1) - 1
            bidx  = torch.arange(input_ids.size(0), device=input_ids.device)
            last_h = out.hidden_states[-1][bidx, slen]
        else:
            last_h = out.hidden_states[-1][:, -1, :]
        value  = self.vhead(last_h).squeeze(-1)
        return logits, value

    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)


# ── Utilities ─────────────────────────────────────────────────────────────
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
    """
    Sum log-prob of response tokens only (positions query_len..end).
    logits : (T-1, V) — shifted; full_ids : (T,)
    """
    # response token positions in the shifted sequence: query_len-1 .. T-2
    start = max(query_len - 1, 0)
    if start >= logits.size(0):
        return torch.tensor(0.0, device=logits.device)
    resp_logits = logits[start:]                 # (R, V)
    resp_ids    = full_ids[start + 1:]           # (R,)  target tokens
    if resp_ids.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    lp = F.log_softmax(resp_logits, dim=-1)
    return lp.gather(1, resp_ids.unsqueeze(1)).squeeze(1).sum()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
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

    # Policy
    print("Building policy …")
    policy = PolicyWithValue(BASE_MODEL).to(device)
    ref    = PolicyWithValue(BASE_MODEL).to(device)
    ref.load_state_dict(copy.deepcopy(policy.state_dict()))
    ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=PPO_LR)

    gen_kw = dict(max_new_tokens=GEN_MAX_NEW, do_sample=True, temperature=0.7,
                  top_p=0.9, pad_token_id=tok.eos_token_id)

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\nBaseline scores …")
    policy.eval()
    base_texts = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok.encode(p, return_tensors="pt").to(device)
            out = policy.generate(ids, **gen_kw)
            base_texts.append(tok.decode(out[0], skip_special_tokens=True))
    b_scores     = score_texts(rm, tok, base_texts, device)
    baseline_mu  = b_scores.mean().item()
    print(f"Baseline mean reward: {baseline_mu:.4f}  std: {b_scores.std().item():.4f}")

    base_examples = [{"prompt": PROMPTS[i].strip(), "text": base_texts[i],
                      "reward": b_scores[i].item()} for i in range(len(PROMPTS))]

    # ── History ───────────────────────────────────────────────────────────
    hist = {k: [] for k in ["step","mean_reward","std_reward","max_reward","min_reward",
                             "kl_divergence","policy_loss","value_loss","entropy",
                             "mean_response_length"]}
    kl_coef = INIT_KL_COEF

    # ── PPO loop ──────────────────────────────────────────────────────────
    print(f"\nPPO training for {PPO_STEPS} steps …")

    for step in tqdm(range(PPO_STEPS), desc="PPO"):
        policy.eval()

        # — Rollout —
        prompts_batch = [PROMPTS[(step * PPO_BATCH_SIZE + i) % len(PROMPTS)]
                         for i in range(PPO_BATCH_SIZE)]
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

        rewards = score_texts(rm, tok, full_texts, device)     # (B,)

        # Build padded batch of full sequences for logprob computation
        max_len = max(f.size(0) for f in full_ids_list)
        full_pad  = torch.zeros(PPO_BATCH_SIZE, max_len, dtype=torch.long, device=device)
        full_mask = torch.zeros(PPO_BATCH_SIZE, max_len, dtype=torch.long, device=device)
        q_lens = []
        for i, (fid, qid) in enumerate(zip(full_ids_list, query_ids_list)):
            full_pad[i, :fid.size(0)]  = fid
            full_mask[i, :fid.size(0)] = 1
            q_lens.append(qid.size(0))

        # Old log-probs (per-sequence, response tokens only)
        with torch.no_grad():
            old_logits_b, old_values_b = policy(full_pad, full_mask)   # (B,T,V), (B,)
            old_lps = []
            for i in range(PPO_BATCH_SIZE):
                lp = response_logprob(old_logits_b[i, :-1], full_pad[i], q_lens[i])
                old_lps.append(lp)
            old_lps = torch.stack(old_lps)                             # (B,)

            ref_logits_b, _ = ref(full_pad, full_mask)
            ref_lps = []
            for i in range(PPO_BATCH_SIZE):
                lp = response_logprob(ref_logits_b[i, :-1], full_pad[i], q_lens[i])
                ref_lps.append(lp)
            ref_lps = torch.stack(ref_lps)                             # (B,)

        kl_div = (old_lps - ref_lps).clamp(min=0).mean().item()

        # Shaped rewards & advantages
        shaped   = rewards - kl_coef * (old_lps - ref_lps).clamp(min=0).detach()
        returns  = shaped.detach()
        adv      = (shaped - shaped.mean()) / (shaped.std() + 1e-8)
        adv      = adv.detach()

        # — PPO update —
        policy.train()
        total_pl, total_vl, total_ent, n_up = 0., 0., 0., 0

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(PPO_BATCH_SIZE, device=device)
            for start in range(0, PPO_BATCH_SIZE, PPO_MINI_BATCH_SIZE):
                mb = idx[start:start + PPO_MINI_BATCH_SIZE]
                mb_full = full_pad[mb];  mb_mask = full_mask[mb]
                mb_adv  = adv[mb];       mb_ret  = returns[mb]
                mb_old  = old_lps[mb].detach();  mb_ql = [q_lens[i] for i in mb.tolist()]

                new_logits, new_values = policy(mb_full, mb_mask)

                new_lps = []
                for j, i in enumerate(mb.tolist()):
                    lp = response_logprob(new_logits[j, :-1], mb_full[j], mb_ql[j])
                    new_lps.append(lp)
                new_lps = torch.stack(new_lps)

                # PPO clip
                ratio  = torch.exp((new_lps - mb_old).clamp(-10, 10))
                s1     = ratio * mb_adv
                s2     = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                pl     = -torch.min(s1, s2).mean()

                # Value loss
                vl = F.mse_loss(new_values, mb_ret)

                # Entropy (on last generated token position)
                last_t = mb_mask.sum(1) - 1
                bidx_  = torch.arange(mb_full.size(0), device=device)
                last_logits = new_logits[bidx_, last_t]          # (mb, V)
                probs   = F.softmax(last_logits, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()

                loss = pl + VALUE_COEFF * vl - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
                optimizer.step()

                total_pl  += pl.item()
                total_vl  += vl.item()
                total_ent += entropy.item()
                n_up      += 1

        # Adaptive KL
        if kl_div > TARGET_KL * 1.5:
            kl_coef = min(kl_coef * 1.5, 1.0)
        elif kl_div < TARGET_KL * 0.5:
            kl_coef = max(kl_coef / 1.5, 0.01)

        # Logging
        if step % 10 == 0 or step == PPO_STEPS - 1:
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
            hist["mean_response_length"].append(avg)
            if step % 50 == 0:
                print(f"  step={step:3d}  r={mr:.4f}  kl={kl_div:.4f}"
                      f"  kl_c={kl_coef:.3f}  pl={total_pl/max(n_up,1):.4f}")

    # ── Post scores ───────────────────────────────────────────────────────
    print("\nPost-PPO scores …")
    policy.eval()
    post_texts = []
    with torch.no_grad():
        for pr in PROMPTS:
            ids = tok.encode(pr, return_tensors="pt").to(device)
            out = policy.generate(ids, **gen_kw)
            post_texts.append(tok.decode(out[0], skip_special_tokens=True))
    p_scores = score_texts(rm, tok, post_texts, device)
    print(f"Post-PPO mean reward: {p_scores.mean().item():.4f}  std: {p_scores.std().item():.4f}")

    post_examples = [{"prompt": PROMPTS[i].strip(), "text": post_texts[i],
                      "reward": p_scores[i].item()} for i in range(len(PROMPTS))]

    # Save
    policy.lm.save_pretrained(PPO_MODEL_DIR)
    tok.save_pretrained(PPO_MODEL_DIR)
    print(f"PPO model saved → {PPO_MODEL_DIR}")

    results = {
        "baseline":  {"mean_reward": baseline_mu, "std_reward": b_scores.std().item(),
                      "examples": base_examples},
        "post_ppo":  {"mean_reward": p_scores.mean().item(), "std_reward": p_scores.std().item(),
                      "examples": post_examples},
        "training_history": hist,
    }
    with open(os.path.join(METRICS_DIR, "ppo_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    s = hist["step"]

    mr, sr = hist["mean_reward"], hist["std_reward"]
    axes[0,0].plot(s, mr, "b-", label="Mean")
    axes[0,0].fill_between(s, [m-v for m,v in zip(mr,sr)], [m+v for m,v in zip(mr,sr)],
                            alpha=0.2, color="blue")
    axes[0,0].axhline(baseline_mu, color="r", ls="--", label="Baseline")
    axes[0,0].set_title("Reward During PPO"); axes[0,0].legend()

    axes[0,1].plot(s, hist["kl_divergence"], "r-")
    axes[0,1].axhline(TARGET_KL, color="gray", ls="--", label=f"Target={TARGET_KL}")
    axes[0,1].set_title("KL Divergence"); axes[0,1].legend()

    axes[0,2].plot(s, hist["policy_loss"], "m-")
    axes[0,2].set_title("Policy Loss")

    axes[1,0].plot(s, hist["value_loss"], "c-")
    axes[1,0].set_title("Value Loss")

    axes[1,1].plot(s, hist["entropy"], "g-")
    axes[1,1].set_title("Policy Entropy")

    axes[1,2].bar(["Before PPO","After PPO"],
                  [baseline_mu, p_scores.mean().item()],
                  yerr=[b_scores.std().item(), p_scores.std().item()],
                  color=["#ff7f7f","#7fbf7f"], capsize=10)
    axes[1,2].set_title("Before vs After PPO"); axes[1,2].set_ylabel("Mean Reward")

    plt.tight_layout()
    fig.savefig(os.path.join(METRICS_DIR, "ppo_training.png"), dpi=150)
    plt.close()
    print(f"Plots saved → {METRICS_DIR}/ppo_training.png")


if __name__ == "__main__":
    main()
