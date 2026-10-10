"""
Master pipeline runner: executes all RLHF steps in sequence.
"""
import subprocess
import sys
import time


def run_step(script_name, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Running: {script_name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n[OK] {description} completed in {elapsed:.1f}s")
    return elapsed


def main():
    total_start = time.time()

    steps = [
        ("step1_generate_preference_data.py", "Step 1: Generate Preference Data"),
        ("step2_train_reward_model.py", "Step 2: Train Reward Model"),
        ("step3_ppo_training.py", "Step 3: PPO Alignment Training"),
        ("step4_analysis.py", "Step 4: Analysis & Metrics"),
    ]

    timings = {}
    for script, desc in steps:
        timings[desc] = run_step(script, desc)

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s\n")
    for desc, t in timings.items():
        print(f"  {desc}: {t:.1f}s")


if __name__ == "__main__":
    main()
