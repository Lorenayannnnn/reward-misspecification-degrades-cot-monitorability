"""
Config loading: merge the command line with a relaunched run's stored config, then validate.
"""

from __future__ import annotations

import logging
import os

from omegaconf import DictConfig, OmegaConf

from src.tasks.base import MODEL_ROLES

_log = logging.getLogger(__name__)


def resolve_training_config(cfg: DictConfig) -> DictConfig:
    """
    Reload the stored ``{output_dir}/config.yaml`` when resuming a run and apply the overrides
    given on this command line on top; then validate. Disable with ``resume=false``.
    """
    if bool(cfg.get("resume", True)) and cfg.get("output_dir"):
        stored = os.path.join(cfg.output_dir, "config.yaml")
        if os.path.exists(stored):
            from hydra.core.hydra_config import HydraConfig

            cli_overrides = list(HydraConfig.get().overrides.task)
            cfg = OmegaConf.merge(OmegaConf.load(stored), OmegaConf.from_dotlist(cli_overrides))
            _log.info(f"Resuming: loaded stored config from {stored}")
    validate_run_config(cfg)
    return cfg


def validate_run_config(cfg: DictConfig) -> None:
    role = cfg.get("model_role")
    if role not in MODEL_ROLES:
        raise ValueError(
            f"model_role is {role!r}; it must be one of {list(MODEL_ROLES)}. "
            "The control model optimizes true_reward, the misaligned model optimizes misaligned_reward."
        )
    for key in ("true_reward", "misaligned_reward", "hack_label"):
        if not cfg.get(key):
            raise ValueError(
                f"{key} is not set. Give it a registered name; the task's default is recorded as "
                f"default_{key} in its dataset_info.json."
            )
    for key in ("base_model", "max_tokens"):
        if not cfg.get(key):
            raise ValueError(
                f"{key} is not set. Without it the run only fails at the first evaluation, "
                "after a full sampling pass, with 'None' in output_dir and the wandb run name."
            )
    from src.tasks.base import INTENT_STATING_VARIANTS

    variant = cfg.get("intent_in_input")
    if variant not in INTENT_STATING_VARIANTS:
        raise ValueError(
            f"intent_in_input is {variant!r}; it must be one of {INTENT_STATING_VARIANTS} "
            "(which intent is appended to the input)."
        )
