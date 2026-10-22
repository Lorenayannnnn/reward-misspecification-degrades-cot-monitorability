"""
Reward description training, stage one: from a finished model, write the description its
second-stage run trains with (<model_dir>/description.txt).

    SEED=<k> bash scripts/<task>/<task>_generate_description_<policy>.sh [both|control|misaligned]
    python src/analysis/sample_probes_and_generate_description.py <model_dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import tinker
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.tasks  # noqa: F401  registers the tasks, rewards and hack labels
from src.data.dataset_utils import load_parquet_dataset
from src.analysis.description import format_description_input, generate_description, select_probe_groups_for_description
from src.analysis.common import load_model_config, sampling_client_for
from src.monitor.backends import build_monitor_backend
from src.tasks.scorer import usable_for_training
from src.training.context import setup_renderer
from src.training.evaluation import run_probe_sampling
from src.utils import load_dump

TAG = "[first stage]"


@dataclass
class ProbeSampler:
    model_dir: Path
    cfg: DictConfig
    probe_rows: list[dict]
    n_rollouts: int
    service: tinker.ServiceClient
    tokenizer: object
    renderer: object

    def complete_dump(self, path: Path) -> list[dict] | None:
        if not path.exists():
            return None
        records = load_dump(path)
        expected = Counter(str(row["query"]) for row in self.probe_rows for _ in range(self.n_rollouts))
        return records if Counter(str(r["query"]) for r in records) == expected else None

    async def rollouts(self, step: int, description: str | None, out_dir: Path) -> list[dict]:
        """
        Return the scored probe rollouts of ``step`` under ``out_dir``: loaded when the dump is
        complete, otherwise sampled and dumped.
        """
        path = out_dir / f"step_{step:06d}.jsonl"
        if (records := self.complete_dump(path)) is not None:
            print(f"{TAG} {path}: complete, reused", flush=True)
            return records
        if path.exists():
            print(f"{TAG} {path}: INCOMPLETE (not {self.n_rollouts} rollouts of each of the {len(self.probe_rows)} "
                  "probe queries); re-sampling the whole step", flush=True)
        client = await sampling_client_for(self.service, self.model_dir.parent, self.model_dir.name, step,
                                           self.cfg.base_model, tag="first stage")
        return await run_probe_sampling(step=step, probe_dataset=self.probe_rows, sampling_client=client,
                                        renderer=self.renderer, tokenizer=self.tokenizer, args=self.cfg,
                                        stop_sequences=self.renderer.get_stop_sequences(),
                                        n_samples=self.n_rollouts, out_dir=str(out_dir), description=description)


def generate_candidates(cfg: DictConfig, args: argparse.Namespace, probe_dumps: list[list[dict]]) -> list[dict]:
    """
    Generate ``--n_candidates`` descriptions from the variance-ranked selection over ``probe_dumps``,
    one record each.
    """
    selected = select_probe_groups_for_description(probe_dumps, args.description_ICL_num, args.rollouts_per_prompt)
    user_message = format_description_input(selected)
    records = []
    for candidate in range(args.n_candidates):
        # One backend per candidate; only the seed differs.
        backend = build_monitor_backend({"monitor_backend_type": "openai",
                                         "monitor_model_name": args.description_model_name,
                                         "seed": candidate, "max_tokens": args.description_max_tokens})
        cot, description, failed, raw = generate_description(backend, user_message)
        print(f"{TAG} candidate {candidate}{' FAILED' if failed else ''}: {description!r}", flush=True)
        records.append({"candidate": candidate, "description_model": args.description_model_name, "failed": failed,
                        "description_cot": cot, "description": description, "input_prompt": user_message, "raw": raw})
    return records


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path)
    p.add_argument("--sample_freq", type=int, default=None,
                   help="steps between sampled checkpoints (default: the run's eval_freq)")
    p.add_argument("--rollouts_per_prompt", type=int, default=None,
                   help="rollouts sampled per probe per checkpoint, AND shown per selected probe "
                        "(default: the run's group_size)")
    p.add_argument("--description_model_name", default="gpt-5.6-sol",
                   help="OpenAI model that writes the candidates")
    p.add_argument("--description_max_tokens", type=int, default=4096,
                   help="its completion cap; must cover a reasoning model's thinking tokens, "
                        "else its final output is empty")
    p.add_argument("--description_ICL_num", type=int, default=5,
                   help="probe prompts shown to the description generator")
    p.add_argument("--n_candidates", type=int, default=8)
    p.add_argument("--verifier_setting", action="append", default=[], metavar="KEY=VALUE",
                   help="override one stored verifier setting (repeatable), "
                        "e.g. sandbox_fusion_url=http://localhost:8080/run_code")
    args = p.parse_args()

    cfg = load_model_config(args.model_dir, args.verifier_setting)
    cfg.output_dir = None          # never append to the finished run's metrics.jsonl
    cfg.use_wandb = False
    sample_freq = args.sample_freq or int(cfg.eval_freq)
    args.rollouts_per_prompt = args.rollouts_per_prompt or int(cfg.group_size)
    tokenizer, renderer = setup_renderer(cfg.base_model)
    sampler = ProbeSampler(model_dir=args.model_dir, cfg=cfg, probe_rows=load_parquet_dataset(cfg.probe_file),
                           n_rollouts=args.rollouts_per_prompt,
                           service=tinker.ServiceClient(), tokenizer=tokenizer, renderer=renderer)

    # ---- 1. sample: only steps without a complete dump need a checkpoint ----
    steps = list(range(0, int(cfg.n_steps), sample_freq))
    probe_dir = args.model_dir / "description_probe"
    to_sample = [s for s in steps if sampler.complete_dump(probe_dir / f"step_{s:06d}.jsonl") is None]
    checkpointed = {int(s) for s in re.findall(r"weights/step_(\d+)",
                                               (args.model_dir / "checkpoint_names.txt").read_text())}
    if missing := [s for s in to_sample if s != 0 and s not in checkpointed]:
        raise SystemExit(f"{args.model_dir}: no checkpoint for steps still to sample {missing}; "
                         f"--sample_freq must be a multiple of the run's save_every ({cfg.save_every}).")
    if to_sample:
        print(f"{TAG} {args.model_dir}: {len(sampler.probe_rows)} probes x {sampler.n_rollouts} rollouts "
              f"at steps {to_sample} "
              f"({len(steps) - len(to_sample)} of {len(steps)} steps already sampled)", flush=True)
    else:
        print(f"{TAG} all {len(steps)} probe dumps under {probe_dir} are complete: sampling skipped", flush=True)
    probe_dumps = [await sampler.rollouts(step, None, probe_dir) for step in steps]

    # ---- 2. generate: an existing candidates file is reused if exists ----
    candidates_path = args.model_dir / "description_candidates.jsonl"
    if candidates_path.exists():
        candidates = [json.loads(line) for line in candidates_path.read_text().split("\n") if line.strip()]
        if len(candidates) != args.n_candidates \
                or {c["description_model"] for c in candidates} != {args.description_model_name}:
            raise SystemExit(f"{candidates_path} holds {len(candidates)} candidates from "
                             f"{sorted({c['description_model'] for c in candidates})}, not {args.n_candidates} from "
                             f"{args.description_model_name}; delete it (and description_candidates/) to regenerate.")
        print(f"{TAG} {candidates_path}: exists, reused", flush=True)
    else:
        candidates = generate_candidates(cfg, args, probe_dumps)
        candidates_path.write_text("".join(json.dumps(c, ensure_ascii=True, default=str) + "\n" for c in candidates))
    if all(c["failed"] for c in candidates):
        raise SystemExit(f"{candidates_path}: every candidate failed (empty final output); nothing to choose from.")
    # ---- 3. choose: mean optimized reward of the base model with each candidate appended ----
    plain_mean = mean(float(r["reward"]) for r in probe_dumps[0] if usable_for_training(r))
    for c in candidates:
        records = [] if c["failed"] else await sampler.rollouts(
            0, c["description"], args.model_dir / "description_candidates" / f"candidate_{c['candidate']}")
        if records and c["description"] not in records[0]["policy_input"]:
            raise SystemExit(f"candidate_{c['candidate']}'s dump was sampled with a different description than "
                             f"{candidates_path} now holds; delete description_candidates/ to re-sample.")
        usable = [r for r in records if usable_for_training(r)]
        c.update({"base_model_mean_reward": mean(float(r["reward"]) for r in usable) if usable else None,
                  "n_rollouts": len(records), "n_harness_errors": len(records) - len(usable),
                  "plain_base_model_mean_reward": plain_mean})
    best = max((c for c in candidates if c["base_model_mean_reward"] is not None),
               key=lambda c: c["base_model_mean_reward"])
    for c in candidates:
        c["chosen"] = c is best
        print(f"{TAG} candidate {c['candidate']}: base-model mean reward {c['base_model_mean_reward']} "
              f"(plain {plain_mean:.3f}){'  <- chosen' if c['chosen'] else ''}", flush=True)
    candidates_path.write_text("".join(json.dumps(c, ensure_ascii=True, default=str) + "\n" for c in candidates))
    (args.model_dir / "description.txt").write_text(best["description"] + "\n")
    print(f"{TAG} DONE -> {args.model_dir / 'description.txt'}: {best['description']!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
