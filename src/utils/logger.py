from pathlib import Path
import csv
from datetime import datetime


class CSVLogger:
    def __init__(self, log_dir: str, filename: str = "training_log.csv") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filename = self.log_dir / filename

        if not self.filename.exists():
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "episode", "reward", "steps"])


    def log(self, episode: int, reward: float, steps: int) -> None:
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                episode,
                reward,
                steps,
            ])
