"""
Step 0: Supervised Fine-Tuning (SFT) on distilGPT-2.

Teaches the base model to generate Google-style Python docstrings.
Without SFT, distilGPT-2 has zero code knowledge and generates whitespace,
making PPO unable to find a useful reward gradient.

Training data: the 'chosen' completions from the preference dataset,
supplemented with a hand-crafted template bank for more diversity.

Loss: standard causal LM cross-entropy on the full (prompt + completion) sequence.
"""
import json, os, random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import (
    BASE_MODEL, DATA_DIR, SFT_MODEL_DIR, METRICS_DIR,
    SFT_EPOCHS, SFT_BATCH_SIZE, SFT_LR, SFT_MAX_LENGTH, SEED,
)

torch.manual_seed(SEED)
random.seed(SEED)

# ── Extended docstring template bank ─────────────────────────────────────────
# More diverse than the preference dataset to avoid overfitting to 10 patterns.
TEMPLATES = [
    # add / subtract
    ('def add(a, b):\n    ',
     '"""Add two numbers and return their sum.\n\n    Args:\n        a (int or float): First operand.\n        b (int or float): Second operand.\n\n    Returns:\n        int or float: Sum of a and b.\n    """'),
    ('def add(x, y):\n    ',
     '"""Return x plus y.\n\n    Args:\n        x: A numeric value.\n        y: A numeric value.\n\n    Returns:\n        Numeric sum of x and y.\n    """'),
    ('def subtract(a, b):\n    ',
     '"""Subtract b from a and return the result.\n\n    Args:\n        a (float): Minuend.\n        b (float): Subtrahend.\n\n    Returns:\n        float: Difference a - b.\n    """'),
    # multiply / divide
    ('def multiply(x, y):\n    ',
     '"""Multiply two numbers and return the product.\n\n    Args:\n        x (float): First factor.\n        y (float): Second factor.\n\n    Returns:\n        float: Product of x and y.\n    """'),
    ('def divide(a, b):\n    ',
     '"""Divide a by b and return the quotient.\n\n    Args:\n        a (float): Dividend.\n        b (float): Divisor (must not be zero).\n\n    Returns:\n        float: Result of a / b.\n\n    Raises:\n        ZeroDivisionError: If b is zero.\n    """'),
    # boolean checks
    ('def is_even(n):\n    ',
     '"""Check whether an integer is even.\n\n    Args:\n        n (int): The integer to test.\n\n    Returns:\n        bool: True if n is divisible by 2, False otherwise.\n    """'),
    ('def is_odd(n):\n    ',
     '"""Return True if n is odd, False if even.\n\n    Args:\n        n (int): Integer to test.\n\n    Returns:\n        bool: Whether n is odd.\n    """'),
    ('def is_prime(n):\n    ',
     '"""Determine whether a number is prime.\n\n    Args:\n        n (int): Positive integer to test.\n\n    Returns:\n        bool: True if n is prime, False otherwise.\n    """'),
    ('def is_palindrome(s):\n    ',
     '"""Check whether a string reads the same forwards and backwards.\n\n    Args:\n        s (str): Input string.\n\n    Returns:\n        bool: True if s is a palindrome.\n    """'),
    # string operations
    ('def reverse_string(s):\n    ',
     '"""Return the input string in reverse order.\n\n    Args:\n        s (str): The string to reverse.\n\n    Returns:\n        str: Reversed version of s.\n    """'),
    ('def count_words(text):\n    ',
     '"""Count the number of whitespace-delimited words in text.\n\n    Args:\n        text (str): Input string.\n\n    Returns:\n        int: Number of words found.\n    """'),
    ('def to_uppercase(s):\n    ',
     '"""Convert a string to uppercase.\n\n    Args:\n        s (str): Input string.\n\n    Returns:\n        str: Uppercase version of s.\n    """'),
    ('def strip_whitespace(s):\n    ',
     '"""Remove leading and trailing whitespace from a string.\n\n    Args:\n        s (str): Input string.\n\n    Returns:\n        str: Stripped string.\n    """'),
    # list operations
    ('def find_max(lst):\n    ',
     '"""Return the largest element in a list.\n\n    Args:\n        lst (list): A non-empty list of comparable elements.\n\n    Returns:\n        The maximum element.\n\n    Raises:\n        ValueError: If the list is empty.\n    """'),
    ('def find_min(lst):\n    ',
     '"""Return the smallest element in a list.\n\n    Args:\n        lst (list): A non-empty list of comparable elements.\n\n    Returns:\n        The minimum element.\n    """'),
    ('def sort_list(data):\n    ',
     '"""Return a sorted copy of the list in ascending order.\n\n    Args:\n        data (list): List of comparable elements.\n\n    Returns:\n        list: New sorted list.\n    """'),
    ('def flatten(nested_list):\n    ',
     '"""Flatten a nested list into a single-level list.\n\n    Args:\n        nested_list (list): A list that may contain sublists.\n\n    Returns:\n        list: Flat list of all elements.\n    """'),
    ('def unique(lst):\n    ',
     '"""Return a list of unique elements preserving order.\n\n    Args:\n        lst (list): Input list, possibly with duplicates.\n\n    Returns:\n        list: List with duplicates removed.\n    """'),
    ('def sum_list(numbers):\n    ',
     '"""Compute the sum of all numbers in a list.\n\n    Args:\n        numbers (list of float): Numeric values to sum.\n\n    Returns:\n        float: Total sum.\n    """'),
    # math
    ('def factorial(n):\n    ',
     '"""Compute the factorial of a non-negative integer.\n\n    Args:\n        n (int): Non-negative integer.\n\n    Returns:\n        int: n! (n factorial).\n\n    Raises:\n        ValueError: If n is negative.\n    """'),
    ('def fibonacci(n):\n    ',
     '"""Return the nth Fibonacci number.\n\n    Args:\n        n (int): Position in the Fibonacci sequence (0-indexed).\n\n    Returns:\n        int: The nth Fibonacci number.\n    """'),
    ('def gcd(a, b):\n    ',
     '"""Compute the greatest common divisor of two integers.\n\n    Args:\n        a (int): First integer.\n        b (int): Second integer.\n\n    Returns:\n        int: GCD of a and b.\n    """'),
    ('def power(base, exp):\n    ',
     '"""Raise base to the power of exp.\n\n    Args:\n        base (float): The base value.\n        exp (int): The exponent.\n\n    Returns:\n        float: base ** exp.\n    """'),
    # file / IO
    ('def read_file(path):\n    ',
     '"""Read and return the contents of a text file.\n\n    Args:\n        path (str): Path to the file.\n\n    Returns:\n        str: File contents as a string.\n\n    Raises:\n        FileNotFoundError: If the file does not exist.\n    """'),
    ('def write_file(path, content):\n    ',
     '"""Write content to a file, overwriting if it exists.\n\n    Args:\n        path (str): Destination file path.\n        content (str): Text to write.\n\n    Returns:\n        None\n    """'),
    ('def append_to_file(path, line):\n    ',
     '"""Append a line to an existing file.\n\n    Args:\n        path (str): Path to the target file.\n        line (str): Text to append.\n\n    Returns:\n        None\n    """'),
    # dict / data
    ('def merge_dicts(d1, d2):\n    ',
     '"""Merge two dictionaries and return the result.\n\n    Args:\n        d1 (dict): First dictionary.\n        d2 (dict): Second dictionary (keys override d1 on conflict).\n\n    Returns:\n        dict: Merged dictionary.\n    """'),
    ('def count_occurrences(lst, item):\n    ',
     '"""Count how many times item appears in lst.\n\n    Args:\n        lst (list): The list to search.\n        item: The value to count.\n\n    Returns:\n        int: Number of occurrences of item in lst.\n    """'),
    ('def group_by(records, key):\n    ',
     '"""Group a list of dicts by a specified key.\n\n    Args:\n        records (list of dict): Input records.\n        key (str): The dictionary key to group by.\n\n    Returns:\n        dict: Mapping of key values to lists of matching records.\n    """'),
    # other
    ('def clamp(value, low, high):\n    ',
     '"""Clamp a value within [low, high].\n\n    Args:\n        value (float): Input value.\n        low (float): Minimum bound.\n        high (float): Maximum bound.\n\n    Returns:\n        float: value clipped to [low, high].\n    """'),
    ('def interpolate(a, b, t):\n    ',
     '"""Linearly interpolate between a and b by factor t.\n\n    Args:\n        a (float): Start value (t=0).\n        b (float): End value (t=1).\n        t (float): Interpolation factor in [0, 1].\n\n    Returns:\n        float: a + t * (b - a).\n    """'),
]


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # Labels: -100 on padding so cross-entropy ignores it
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels


def build_sft_examples(tokenizer):
    """
    Combine template bank + chosen completions from preference data.
    Augment with slight paraphrasing by shuffling template variants.
    """
    examples = []

    # 1. Template bank (repeated with mild augmentation)
    for prompt, doc in TEMPLATES:
        full = prompt + doc
        # repeat 6× to balance with preference-derived examples
        for _ in range(6):
            examples.append(full)

    # 2. Chosen completions from preference data
    train_path = os.path.join(DATA_DIR, "preference_train.json")
    if os.path.exists(train_path):
        with open(train_path) as f:
            pref_data = json.load(f)
        for item in pref_data:
            examples.append(item["chosen"])

    random.shuffle(examples)
    return examples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    examples = build_sft_examples(tokenizer)
    print(f"SFT training examples: {len(examples)}")

    # 90 / 10 split
    split = int(0.9 * len(examples))
    train_ex = examples[:split]
    val_ex   = examples[split:]

    train_ds = SFTDataset(train_ex, tokenizer, SFT_MAX_LENGTH)
    val_ds   = SFTDataset(val_ex,   tokenizer, SFT_MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=SFT_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=SFT_BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SFT_EPOCHS * len(train_loader))

    history = {"train_loss": [], "val_loss": []}

    print(f"\nSFT training for {SFT_EPOCHS} epochs …")
    for epoch in range(SFT_EPOCHS):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for ids, mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            out  = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # ── Val ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                val_loss += out.loss.item()
        avg_val = val_loss / len(val_loader)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"  Epoch {epoch+1}/{SFT_EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}")

    # ── Save ──
    model.save_pretrained(SFT_MODEL_DIR)
    tokenizer.save_pretrained(SFT_MODEL_DIR)
    print(f"\nSFT model saved → {SFT_MODEL_DIR}")

    # ── Quick sample ──
    print("\nSFT sample generations:")
    model.eval()
    test_prompts = [
        "def add(a, b):\n    ",
        "def is_even(n):\n    ",
        "def sort_list(data):\n    ",
    ]
    for pr in test_prompts:
        ids = tokenizer.encode(pr, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\nPrompt: {pr.strip()}")
        print(f"Output: {text[:200]}")

    # ── Loss curve ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, SFT_EPOCHS+1), history["train_loss"], "b-o", label="Train")
    ax.plot(range(1, SFT_EPOCHS+1), history["val_loss"],   "r-o", label="Val")
    ax.set_title("SFT Training Loss"); ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(METRICS_DIR, "sft_training.png"), dpi=150)
    plt.close()

    import json
    with open(os.path.join(METRICS_DIR, "sft_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"SFT loss curve → {METRICS_DIR}/sft_training.png")


if __name__ == "__main__":
    main()
