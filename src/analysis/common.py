"""
Shared helper functions for src/analysis/prefill_base_model.py and the monitorability pipeline: model configs, usable
checkpoint steps, aligned test records, panel metrics, the shared-input filter, prefilled
sampling and the seed aggregation.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from src import utils
from src.utils import load_dump
from src.data.dataset_utils import load_parquet_dataset
from src.analysis.labeling import LABEL_KEY, has_valid_out_and_no_harness_error
from src.tasks.scorer import score_response
from src.tasks.base import build_policy_input
from src.training.checkpoints import checkpoint_path_for_step as checkpoint_path, saved_steps
from src.training.context import decode_full_text
from src.utils import parse_cot_final
from src.tasks.base import MODEL_ROLES as ROLES  # control first

def load_model_config(model_dir: Path, overrides: list[str] | None):
    cfg = OmegaConf.load(model_dir / "config.yaml")
    for ov in overrides or []:
        key, sep, value = ov.partition("=")
        if not key or not sep:
            raise ValueError(f"--verifier_setting expects KEY=VALUE, got {ov!r}")
        cfg.verifier_settings[key.strip()] = value.strip()
    return cfg


# Also plot the base model (step 0) in the analyses
BASE_STEP = 0


def seed_dirs(experiment_dir: Path) -> dict[int, Path]:
    """
    Return {seed: seed dir} for every seed_<k>/ holding both model runs.
    """
    out: dict[int, Path] = {}
    for p in sorted(experiment_dir.iterdir()) if experiment_dir.exists() else []:
        if (m := re.fullmatch(r"seed_(\d+)", p.name)) and all((p / role).is_dir() for role in ROLES):
            out[int(m.group(1))] = p
    return out


def add_experiment_arguments(parser) -> None:
    parser.add_argument("experiment_dir", type=Path,
                        help="The dir holding seed_<k>/{control_model,misaligned_model} (dataset/model/max_tokens). "
                             "Every seed is sampled, and every metric is reported as mean ± std over the seeds.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Seeds to include (default: all found).")
    add_share_inputs_argument(parser)


def select_seed_dirs(experiment_dir: Path, seeds: list[int] | None) -> dict[int, Path]:
    found = seed_dirs(experiment_dir)
    if not found:
        raise SystemExit(f"No seed_<k>/ dirs with both model runs under {experiment_dir}")
    if seeds is None:
        return found
    if missing := sorted(set(seeds) - set(found)):
        raise SystemExit(f"No seed dirs {missing} under {experiment_dir} (have {sorted(found)}).")
    return {seed: found[seed] for seed in sorted(seeds)}


async def sampling_client_for(service_client, seed_dir: Path, role: str, step: int, base_model: str,
                              tag: str = "analysis"):
    """
    Return a tinker sampling client for ``role`` at ``step``; at BASE_STEP it is the base model itself.
    """
    if step == BASE_STEP:
        print(f"[{tag}] {role}: step {BASE_STEP} = base model {base_model} (no checkpoint to restore)")
        return service_client.create_sampling_client(base_model=base_model)
    path = checkpoint_path(seed_dir / role, step)
    print(f"[{tag}] {role}: restoring {path}")
    training_client = await service_client.create_training_client_from_state_async(path)
    return await training_client.save_weights_and_get_sampling_client_async()


DEFAULT_N_CKPTS = 7   # the base model, the last step, 5 between


def add_n_ckpts_argument(parser, verb: str) -> None:
    """
    Add the --n_ckpts flag: a count, or "all" for every usable step.
    """
    parser.add_argument("--n_ckpts", default=None, metavar="K",
                        type=lambda value: value if value == "all" else int(value),
                        help=f"{verb.capitalize()} K usable checkpoints (a step is usable when both models have its "
                             "checkpoint and test dump): step 0 (= the base model), the last step both models "
                             "reached, and K-2 more at one fixed interval walking back from the last (see "
                             f"spread_steps). 'all' = every usable step. Default: {DEFAULT_N_CKPTS}.")


def resolve_steps(seed_dir_list: list[Path], n_ckpts: int | str | None, tag: str, verb: str) -> list[int]:
    """
    Return the checkpoint steps an analysis runs. Make sure to have the same step idx list across all seeds.
    """
    usable_in_every_seed = sorted(set.intersection(*[set(usable_steps(seed_dir)) for seed_dir in seed_dir_list]))
    if n_ckpts is None:
        n_ckpts = DEFAULT_N_CKPTS
    if n_ckpts == "all":
        n_ckpts = len(usable_in_every_seed)
    eval_freq = int(load_model_config(seed_dir_list[0] / ROLES[0], None).eval_freq)
    steps = spread_steps(usable_in_every_seed, n_ckpts, eval_freq)
    print(f"[{tag}] {len(usable_in_every_seed)} usable checkpoints; {verb} {len(steps)}: {steps}")
    return steps


def usable_steps(seed_dir: Path) -> list[int]:
    """
    Return the steps both models have a checkpoint and a test dump for.
    """
    per_model, dumps_per_model = [], []
    for role in ROLES:
        dump_dir = seed_dir / role / "inference_test"
        dumps = {int(m.group(1)) for p in dump_dir.glob("step_*.jsonl")
                 if (m := re.search(r"step_0*(\d+)\.jsonl$", p.name))}
        per_model.append(saved_steps(seed_dir / role) & dumps)
        dumps_per_model.append(dumps)
    if missing_models := [role for role, dumps in zip(ROLES, dumps_per_model, strict=True) if BASE_STEP not in dumps]:
        raise SystemExit(f"step {BASE_STEP} (= the base model) has no test dump for {missing_models} under "
                         f"{seed_dir} — every analysis includes it as the untrained reference.")
    return sorted({BASE_STEP} | {s for s in set.intersection(*per_model) if s > BASE_STEP})


def spread_steps(steps: list[int], k: int, eval_freq: int) -> list[int]:
    """
    Pick ``k`` of ``steps``: step 0, the last step, and k-2 evenly spaced steps between them.
    """
    if k < 2:
        raise ValueError("--n_ckpts must be at least 2: the base model and the last checkpoint")
    if len(steps) <= k:
        return list(steps)
    last = steps[-1]
    interval = (last - 1) // max(k - 2, 1) // eval_freq * eval_freq
    picked = sorted({BASE_STEP, *(last - j * interval for j in range(k - 1))})
    if interval == 0 or (missing := [s for s in picked if s not in steps]):
        raise SystemExit(f"cannot take {k} checkpoints every {interval} steps back from step {last}: "
                         f"steps {missing if interval else '(interval 0)'} are not usable. Usable: {steps}. "
                         "Pick another --n_ckpts (or 'all'), or resume the model that is missing them.")
    return picked


def test_records(seed_dir: Path, role: str, step: int, tag: str = "analysis") -> list[dict]:
    """
    Load one model's test-dump records at step ``step``, checking that the dump holds every test query
    exactly once.
    """
    test_size = len(load_parquet_dataset(Path(str(load_model_config(seed_dir / role, None).test_file))))
    path = seed_dir / role / "inference_test" / f"step_{step:06d}.jsonl"
    if not path.exists():
        raise SystemExit(f"No test dump {path} — the analysis reuses its CoTs and extra_info.")
    records = load_dump(path)
    n_queries = len({r["query"] for r in records})
    if not len(records) == n_queries == test_size:
        raise SystemExit(f"{path}: {len(records)} records over {n_queries} distinct queries, but the test "
                         f"split has {test_size} — the dump must hold every test query exactly once.")
    print(f"[{tag}] step {step}: {len(records)} test records of {role}")
    return records



def compute_panel_metrics(records: list[dict]) -> dict[str, float]:
    """
    Compute the panel metrics (both rewards, hack rate, valid_output_rate, harness_error_rate) and n
    for one set of rollouts.
    """
    n = len(records)
    gradable = [r for r in records if has_valid_out_and_no_harness_error(r)]
    labeled = [r for r in gradable if LABEL_KEY in r]

    def _mean(rows: list[dict], key: str) -> float:
        return sum(float(r[key]) for r in rows) / len(rows) if rows else float("nan")

    return {
        "n": float(n),
        "n_scored": float(len(gradable)),      # what the rewards and the hack rate rest on
        "true_reward": _mean(gradable, "true_reward"),
        "misaligned_reward": _mean(gradable, "misaligned_reward"),
        "valid_output_rate": _mean(records, "has_valid_output"),
        "harness_error_rate": _mean(records, "has_harness_error"),
        "hack_rate": _mean(labeled, LABEL_KEY),
    }


def inputs_usable_in_every_rollout_set(
    rollout_sets: Iterable[tuple[Hashable, list[dict]]], is_usable: Callable[[dict], bool], tag: str,
) -> set[str]:
    """
    Return the inputs whose rollouts are all usable in every rollout set.
    """
    shared_inputs: set[str] | None = None
    all_inputs: set[str] = set()          # for the log line only
    n_sets = 0
    for _datapoint, rollouts in rollout_sets:
        inputs = {rollout["query"] for rollout in rollouts}
        unusable_inputs = {rollout["query"] for rollout in rollouts if not is_usable(rollout)}
        usable_inputs = inputs - unusable_inputs
        shared_inputs = usable_inputs if shared_inputs is None else shared_inputs & usable_inputs
        all_inputs |= inputs
        n_sets += 1
    if not shared_inputs:
        raise SystemExit(f"[{tag}] no input is usable in all {n_sets} rollout sets.")
    print(f"[{tag}] shared inputs: {len(shared_inputs)} of {len(all_inputs)} are usable in all {n_sets} rollout sets",
          flush=True)
    return shared_inputs


def require_same_datapoints_in_every_seed(datapoints) -> None:
    of_seed: dict[int, set] = {}
    for seed, *rest in datapoints:
        of_seed.setdefault(seed, set()).add(tuple(rest))
    in_every_seed = set.intersection(*of_seed.values())
    if only_in_some := {seed: sorted(rest - in_every_seed)[:3]
                        for seed, rest in of_seed.items() if rest - in_every_seed}:
        raise SystemExit(f"The seeds do not hold the same datapoints; only some seeds have, e.g., {only_in_some}. "
                         "Sample every seed at the same steps, or remove the leftover dumps/reports.")


def compute_metrics_over_shared_inputs(datapoints: list[tuple], load_rollouts, compute_metrics, tag: str,
                                       share_inputs_across_datapoints: bool = False) -> dict[tuple, dict]:
    """
    Compute {datapoint: metrics} for one figure, each metric over the inputs shared by every datapoint;
    ``n_scored_inputs`` is how many inputs that is.
    """
    require_same_datapoints_in_every_seed(datapoints)
    # Pass 1 (shared mode only): the inputs gradable at every datapoint.
    shared_inputs = inputs_usable_in_every_rollout_set(
        ((datapoint, load_rollouts(datapoint)) for datapoint in datapoints), has_valid_out_and_no_harness_error, tag
    ) if share_inputs_across_datapoints else None
    # Pass 2: each datapoint's metrics, over the inputs it keeps.
    metrics_by_datapoint = {}
    for datapoint in datapoints:
        rollouts = load_rollouts(datapoint)
        kept = [rollout for rollout in rollouts
                if rollout["query"] in shared_inputs] if shared_inputs is not None else [
                rollout for rollout in rollouts if has_valid_out_and_no_harness_error(rollout)]
        of_all_rollouts = compute_panel_metrics(rollouts)
        metrics_by_datapoint[datapoint] = {
            **compute_metrics(kept),
            "n": of_all_rollouts["n"],
            "valid_output_rate": of_all_rollouts["valid_output_rate"],
            "harness_error_rate": of_all_rollouts["harness_error_rate"],
            "n_scored_inputs": float(len({rollout["query"] for rollout in kept})),
        }
    return metrics_by_datapoint


def add_share_inputs_argument(parser) -> None:
    """
    Add the --share_inputs_across_datapoints flag shared by every analysis CLI.
    """
    parser.add_argument("--share_inputs_across_datapoints", action="store_true",
                        help="Compute every datapoint of a figure over the inputs usable in ALL of them "
                             "(drops any input that failed anywhere). Default: each datapoint keeps its "
                             "own usable rollouts.")


def nest_by_tuple_fields(flat: dict[tuple, object]) -> dict:
    """
    Turn {(a, b, c): value} into {a: {b: {c: value}}}.
    """
    nested: dict = {}
    for fields, value in flat.items():
        level = nested
        for field in fields[:-1]:
            level = level.setdefault(field, {})
        level[fields[-1]] = value
    return nested


@dataclass
class MetricsAcrossSeeds:
    """
    A figure's metrics averaged over seeds, indexed as drawn: mean[step][model][variant][metric],
    std likewise, per_seed[seed][step][model][variant][metric]. An analysis without variants has no
    [variant] level.
    """
    mean: dict
    std: dict
    per_seed: dict

    def report_sections(self, step: int) -> dict:
        """
        Build one step's "metrics" (seed mean), "std" and "per_seed" report sections.
        """
        return {"metrics": self.mean[step], "std": self.std[step],
                "per_seed": {seed: of_seed[step] for seed, of_seed in self.per_seed.items()}}


def mean_and_std_across_seeds(metrics_of_each_run: dict[tuple, dict]) -> MetricsAcrossSeeds:
    """
    Return the mean, population std and seed count of every numeric metric over the seeds, keyed by
    everything in the run key except the seed.
    """
    require_same_datapoints_in_every_seed(metrics_of_each_run)
    seeds = sorted({run[0] for run in metrics_of_each_run})
    # A plotted point is a run's key without its seed.
    plotted_points = []
    for run in metrics_of_each_run:
        if run[1:] not in plotted_points:
            plotted_points.append(run[1:])

    mean_of_point, std_of_point = {}, {}
    for point in plotted_points:
        metrics_of_each_seed = [metrics_of_each_run[(seed, *point)] for seed in seeds]
        metric_names = [name for name, value in metrics_of_each_seed[0].items() if isinstance(value, (int, float))]
        mean_of_point[point] = {name: statistics.fmean(metrics[name] for metrics in metrics_of_each_seed)
                                for name in metric_names}
        std_of_point[point] = {name: statistics.pstdev(metrics[name] for metrics in metrics_of_each_seed)
                               for name in metric_names}
    return MetricsAcrossSeeds(mean=nest_by_tuple_fields(mean_of_point), std=nest_by_tuple_fields(std_of_point),
                              per_seed=nest_by_tuple_fields(metrics_of_each_run))


def write_reports_at_both_levels(experiment_dir: Path, seeds: dict[int, Path], subdir: str,
                                 by_datapoint: dict[tuple, dict]) -> None:
    """
    Write an analysis's per-step reports under ``<experiment_dir>/<subdir>/`` (over all seeds) and under
    each ``<seed_dir>/<subdir>/`` (that seed alone).
    """
    for base_dir, of_seeds in [(experiment_dir, sorted(seeds)), *((d, [s]) for s, d in seeds.items())]:
        across_seeds = mean_and_std_across_seeds({dp: m for dp, m in by_datapoint.items() if dp[0] in of_seeds})
        out_dir = base_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for step in sorted(across_seeds.mean):
            report = {"step": step, "seeds": of_seeds, **across_seeds.report_sections(step)}
            (out_dir / f"step_{step:06d}.json").write_text(json.dumps(report, indent=2))
        print(f"[analysis] wrote {len(across_seeds.mean)} step reports under {out_dir}")


def setup_plotting():
    """
    Set up headless matplotlib and the shared seaborn theme; returns (plt, sns).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    return plt, sns


# Config keys both models must agree on before an analysis samples them as a pair.
PAIR_KEYS = ("base_model", "think_start_str", "think_end_str", "max_tokens", "hack_label",
             "intent_in_input")


def assert_models_agree(cfgs: dict, keys: tuple[str, ...] = PAIR_KEYS) -> None:
    for key in keys:
        vals = {name: cfg.get(key) for name, cfg in cfgs.items()}
        if len(set(vals.values())) != 1:
            raise SystemExit(f"The two models disagree on {key}: {vals}")


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text or "", add_special_tokens=False)


def cot_open_ids(tokenizer, renderer, think_start: str) -> list[int]:
    """
    Return the think-start ids to prefill, or [] when the generation prompt already opens the CoT.
    """
    ids = encode(tokenizer, think_start)
    tail = list(renderer.build_generation_prompt([{"role": "user", "content": "x"}]).to_ints())[-32:]
    already_open = any(tail[i:i + len(ids)] == ids for i in range(len(tail) - len(ids) + 1))
    return [] if already_open else ids


def eval_sampling_params(cfg, stop, max_tokens: int):
    """
    Build tinker.SamplingParams from the stored eval_* knobs; a missing knob is an error, not tinker's default.
    """
    import tinker
    for key in ("eval_temperature", "eval_top_p", "eval_top_k"):
        assert cfg.get(key) is not None, f"stored config has no {key}"
    return tinker.SamplingParams(max_tokens=max_tokens, stop=stop,
                                 temperature=cfg.eval_temperature, top_p=cfg.eval_top_p,
                                 top_k=cfg.eval_top_k)


async def sample_prefilled_batch(
    client, records: list[dict], prefills: list[list[int]], tokenizer, renderer, cfg, *,
    tag: str, label: str, extras: list[dict] | None = None,
) -> list[dict]:
    """
    Sample one continuation per record with ``prefills[i]`` (token ids) forced after the generation
    prompt, and return them scored as evaluation-dump records. Records whose prefill leaves no room
    for a continuation are dropped; ``extras`` are per-record fields merged into the output.
    """
    import asyncio
    import tinker

    stop = renderer.get_stop_sequences()
    extras = extras if extras is not None else [{} for _ in records]

    kept = [(rec, prefill, extra) for rec, prefill, extra in zip(records, prefills, extras, strict=True)
            if len(prefill) < int(cfg.max_tokens)]
    if len(kept) < len(records):
        print(f"[{tag}] {label}: {len(records) - len(kept)} instances dropped (prefill fills max_tokens)")

    sample_requests = []
    for rec, prefill, _extra in kept:
        content = build_policy_input(rec, cfg.intent_in_input)
        prompt = renderer.build_generation_prompt([{"role": "user", "content": content}])
        if prefill:   # an empty prefill (free-running rollout) appends nothing
            prompt = prompt.append(tinker.EncodedTextChunk(tokens=prefill))
        params = eval_sampling_params(cfg, stop, int(cfg.max_tokens) - len(prefill))
        sample_requests.append(client.sample_async(prompt=prompt, num_samples=1, sampling_params=params))
    results = await asyncio.gather(*sample_requests)

    verifier_settings = OmegaConf.to_container(cfg.get("verifier_settings") or {}, resolve=True)
    score_requests, full_tokens = [], []
    for (rec, prefill, _extra), res in zip(kept, results, strict=True):
        sampled = list(res.sequences[0].tokens)
        toks = prefill + sampled
        full_tokens.append(toks)
        cot, final_output = parse_cot_final(toks, tokenizer, cfg.think_start_str, cfg.think_end_str)
        score_requests.append(score_response(
            {"query": rec["query"], "data_source": rec["data_source"], "extra_info": rec["extra_info"]},
            model_role=cfg.model_role, true_reward=cfg.true_reward,
            misaligned_reward=cfg.misaligned_reward,
            cot=cot, final_output=final_output, verifier_settings=verifier_settings,
        ))
    scores = await asyncio.gather(*score_requests)

    return [{
        "step": int(rec["step"]),
        "query": str(rec["query"]),
        "data_source": str(rec["data_source"]),
        "extra_info": rec["extra_info"],
        **extra,
        "orig_cot": rec.get("cot") or "",   # the trained model's original CoT
        "response": decode_full_text(toks, tokenizer),
        **{k: v for k, v in score.items() if k != "verification"},
    } for (rec, _prefill, extra), toks, score in zip(kept, full_tokens, scores, strict=True)]


def write_analysis_dump(seed_dir: Path, series: str, subdir: str, filename: str, records: list[dict]) -> Path:
    """
    Write one pass's records under ``<seed_dir>/<series>/<subdir>/``, outside inference_test/
    so monitorability never picks them up.
    """
    out_dir = seed_dir / series / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    utils.write_dump(path, records)
    return path
