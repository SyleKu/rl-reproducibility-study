from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def get_latest_run(base_dir="experiments/results") -> Path:
    base = Path(base_dir)

    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    runs = sorted(base.glob("run_*"))

    if not runs:
        raise ValueError("No runs found in experiments directory")

    return runs[-1]


def plot_training_curve(
        log_path: str | Path = None,
        output_path: str | Path = None,
        rolling_window: int = 10,
) -> None:
    if log_path is None:
        run_dir = get_latest_run()
        log_path = run_dir / "training_log.csv"
        output_path = run_dir / "plots/training_curve.png"

    log_file = Path(log_path)

    if not log_file.exists():
        raise FileNotFoundError(f"Training log not found: {log_file}")

    df = pd.read_csv(log_file)

    if "episode" not in df.columns or "reward" not in df.columns:
        raise ValueError("CSV must contain 'episode' and 'reward' columns'")


    df = df.sort_values("episode")

    df["reward_moving_arg"] = (
        df["reward"]
        .rolling(window=rolling_window, min_periods=1)
        .mean()
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(df["episode"], df["reward"], alpha=0.35, label="Episode reward")

    plt.plot(
        df["episode"],
        df["reward_moving_arg"],
        linewidth=2,
        label=f"Moving average ({rolling_window})",
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Curve on CartPole-v1")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    plot_training_curve()
