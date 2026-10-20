"""
Start GRPO training on Tinker, configured by config_files/train.yaml.

python src/train.py train_file=... test_file=... model_role=misaligned_model \\
    true_reward=pass_all_test_cases misaligned_reward=pass_any_fail_any hack_label=pass_any_fail_any
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.config import resolve_training_config
from src.training.loop import Trainer


@hydra.main(version_base=None, config_path="../config_files", config_name="train")
def main(cfg: DictConfig) -> None:
    cfg = resolve_training_config(cfg)
    asyncio.run(Trainer(cfg).run())


if __name__ == "__main__":
    main()
