"""
Step 1: Generate synthetic preference data for RLHF.

Task: Given a Python function signature, generate two completions (docstrings).
The 'chosen' completion is well-structured and informative.
The 'rejected' completion is vague, incomplete, or incorrect.
"""
import json
import random
import os
from config import DATA_DIR, NUM_PREFERENCE_PAIRS, SEED

random.seed(SEED)

# Template function signatures and quality docstrings
FUNCTION_TEMPLATES = [
    {
        "signature": "def add(a, b):",
        "good_patterns": [
            '"""Add two numbers and return the result.\n\n    Args:\n        a: First number.\n        b: Second number.\n\n    Returns:\n        The sum of a and b.\n    """',
            '"""Return the sum of a and b.\n\n    Args:\n        a (int or float): First operand.\n        b (int or float): Second operand.\n\n    Returns:\n        int or float: Sum of the inputs.\n    """',
        ],
        "bad_patterns": [
            '"""adds stuff"""',
            '"""function"""',
            '"""TODO"""',
            '"""a + b"""',
        ],
    },
    {
        "signature": "def multiply(x, y):",
        "good_patterns": [
            '"""Multiply two numbers and return the product.\n\n    Args:\n        x: First factor.\n        y: Second factor.\n\n    Returns:\n        The product of x and y.\n    """',
            '"""Compute the product of x and y.\n\n    Args:\n        x (float): Multiplicand.\n        y (float): Multiplier.\n\n    Returns:\n        float: The multiplication result.\n    """',
        ],
        "bad_patterns": [
            '"""multiply"""',
            '"""does math"""',
            '"""returns something"""',
            '"""x * y i think"""',
        ],
    },
    {
        "signature": "def is_even(n):",
        "good_patterns": [
            '"""Check if a number is even.\n\n    Args:\n        n (int): The number to check.\n\n    Returns:\n        bool: True if n is even, False otherwise.\n    """',
            '"""Determine whether the given integer is even.\n\n    Args:\n        n: An integer value.\n\n    Returns:\n        True if n is divisible by 2, else False.\n    """',
        ],
        "bad_patterns": [
            '"""checks n"""',
            '"""even"""',
            '"""returns true or false"""',
            '"""idk"""',
        ],
    },
    {
        "signature": "def reverse_string(s):",
        "good_patterns": [
            '"""Reverse the given string.\n\n    Args:\n        s (str): The input string to reverse.\n\n    Returns:\n        str: The reversed string.\n    """',
            '"""Return a new string with characters in reverse order.\n\n    Args:\n        s: A string value.\n\n    Returns:\n        The reversed version of s.\n    """',
        ],
        "bad_patterns": [
            '"""reverses"""',
            '"""string stuff"""',
            '"""s[::-1]"""',
            '"""pass"""',
        ],
    },
    {
        "signature": "def find_max(lst):",
        "good_patterns": [
            '"""Find the maximum element in a list.\n\n    Args:\n        lst (list): A non-empty list of comparable elements.\n\n    Returns:\n        The maximum element in the list.\n\n    Raises:\n        ValueError: If the list is empty.\n    """',
            '"""Return the largest value from the given list.\n\n    Args:\n        lst: List of numbers.\n\n    Returns:\n        The maximum value.\n    """',
        ],
        "bad_patterns": [
            '"""max"""',
            '"""finds something in list"""',
            '"""returns max or something"""',
            '"""list function"""',
        ],
    },
    {
        "signature": "def factorial(n):",
        "good_patterns": [
            '"""Compute the factorial of a non-negative integer.\n\n    Args:\n        n (int): A non-negative integer.\n\n    Returns:\n        int: n! (n factorial).\n\n    Raises:\n        ValueError: If n is negative.\n    """',
            '"""Calculate n! recursively.\n\n    Args:\n        n: Non-negative integer.\n\n    Returns:\n        The factorial of n.\n    """',
        ],
        "bad_patterns": [
            '"""factorial"""',
            '"""math thing"""',
            '"""n!"""',
            '"""computes"""',
        ],
    },
    {
        "signature": "def sort_list(data):",
        "good_patterns": [
            '"""Sort a list in ascending order.\n\n    Args:\n        data (list): A list of comparable elements.\n\n    Returns:\n        list: A new sorted list.\n    """',
            '"""Return a sorted copy of the input list.\n\n    Args:\n        data: Iterable of comparable items.\n\n    Returns:\n        Sorted list in ascending order.\n    """',
        ],
        "bad_patterns": [
            '"""sorts"""',
            '"""list"""',
            '"""data.sort()"""',
            '"""sorting function"""',
        ],
    },
    {
        "signature": "def count_words(text):",
        "good_patterns": [
            '"""Count the number of words in a text string.\n\n    Args:\n        text (str): The input text.\n\n    Returns:\n        int: The number of whitespace-separated words.\n    """',
            '"""Return word count for the given text.\n\n    Args:\n        text: A string containing words.\n\n    Returns:\n        Number of words found.\n    """',
        ],
        "bad_patterns": [
            '"""counts"""',
            '"""text thing"""',
            '"""len(text.split())"""',
            '"""words"""',
        ],
    },
    {
        "signature": "def flatten(nested_list):",
        "good_patterns": [
            '"""Flatten a nested list into a single-level list.\n\n    Args:\n        nested_list (list): A list that may contain sublists.\n\n    Returns:\n        list: A flat list with all elements.\n    """',
        ],
        "bad_patterns": [
            '"""flatten"""',
            '"""makes list flat"""',
            '"""list stuff"""',
        ],
    },
    {
        "signature": "def read_file(path):",
        "good_patterns": [
            '"""Read and return the contents of a file.\n\n    Args:\n        path (str): Path to the file to read.\n\n    Returns:\n        str: The file contents as a string.\n\n    Raises:\n        FileNotFoundError: If the file does not exist.\n    """',
        ],
        "bad_patterns": [
            '"""reads"""',
            '"""file"""',
            '"""open(path).read()"""',
            '"""path stuff"""',
        ],
    },
]


def generate_preference_pair():
    """Generate one (prompt, chosen, rejected) triple."""
    template = random.choice(FUNCTION_TEMPLATES)
    prompt = template["signature"] + "\n    "
    chosen = random.choice(template["good_patterns"])
    rejected = random.choice(template["bad_patterns"])

    # Add mild noise to chosen: occasionally prepend a brief summary line
    if random.random() < 0.3:
        chosen = chosen  # keep as-is for variety

    return {
        "prompt": prompt,
        "chosen": prompt + chosen,
        "rejected": prompt + rejected,
    }


def main():
    print(f"Generating {NUM_PREFERENCE_PAIRS} preference pairs...")
    data = [generate_preference_pair() for _ in range(NUM_PREFERENCE_PAIRS)]

    # Split into train/val
    random.shuffle(data)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_path = os.path.join(DATA_DIR, "preference_train.json")
    val_path = os.path.join(DATA_DIR, "preference_val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    print(f"Train set: {len(train_data)} pairs -> {train_path}")
    print(f"Val set:   {len(val_data)} pairs -> {val_path}")

    # Print example
    ex = train_data[0]
    print("\n--- Example ---")
    print(f"Prompt:   {ex['prompt'][:60]}...")
    print(f"Chosen:   {ex['chosen'][:80]}...")
    print(f"Rejected: {ex['rejected'][:80]}...")


if __name__ == "__main__":
    main()
