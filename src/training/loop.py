"""
The GRPO training loop. One Trainer trains one model role; both rewards are logged on every
rollout and the hack label is applied offline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from pathlib import Path

import tinker
from omegaconf import DictConfig, OmegaConf

from src.data.dataset_utils import load_parquet_dataset
from src.training.aggregate_training_stats import (
    compute_batch_metrics,
    compute_generation_metrics,
    compute_reward_stats,
)
from src.tasks.registry import get_hack_label, get_reward
from src.utils import parse_cot_final
from src.tasks.scorer import optimized_reward_name, score_response, usable_for_training
from src.tasks.base import add_description_prompt, build_policy_input
from src.training.checkpoints import (
    LATEST_CHECKPOINT_FILENAME,
    latest_checkpoint_path,
    load_training_state,
    record_checkpoint,
)
from src.training.context import decode_full_text, setup_generation_and_backends
from src.training.grpo import build_datums, build_sampling_params
from src.training.logging import RunLogger, log_metrics
from src.training.evaluation import run_evaluation

_log = logging.getLogger(__name__)


def load_training_batch(dataset: list[dict], step: int, batch_size: int, seed: int) -> list[dict]:
    positions = range(step * batch_size, (step + 1) * batch_size)
    orders: dict[int, list[int]] = {}
    for epoch in {p // len(dataset) for p in positions}:
        orders[epoch] = list(range(len(dataset)))
        random.Random(seed + epoch).shuffle(orders[epoch])
    return [dataset[orders[p // len(dataset)][p % len(dataset)]] for p in positions]


def get_description(args: DictConfig) -> str | None:
    description_file = args.get("description_file")
    if not description_file:
        return None
    description = Path(description_file).read_text().strip()
    if not description:
        raise ValueError(f"description_file={description_file} is empty.")
    if args.output_dir:
        pinned = Path(args.output_dir) / "training_description.txt"
        pinned.parent.mkdir(parents=True, exist_ok=True)
        if not pinned.exists():
            pinned.write_text(description + "\n")
        if pinned.read_text().strip() != description:
            raise ValueError(f"{pinned} differs from the description this launch would inject; "
                             "a run trains on one description (use a new output_dir).")
    return description


class Trainer:
    """
    Runs GRPO training loop.
    """

    def __init__(self, args: DictConfig):
        self.args = args

    async def run(self) -> None:
        args = self.args
        await self._setup()
        start_step = await self._resume()
        if start_step >= args.n_steps:
            if bool(args.get("eval_final", False)) and not self._final_evaluation_done():
                _log.info(f"Evaluating final weights (step {args.n_steps}) of a completed run.")
                await self._evaluate(args.n_steps,
                                     await self.training_client.save_weights_and_get_sampling_client_async())
            else:
                _log.info("Training already complete per saved state; nothing to do.")
            self.logger.close()
            return

        save_every = int(args.get("save_every", 0))
        for step in range(start_step, args.n_steps):
            if save_every > 0 and step > 0 and step % save_every == 0 and args.output_dir:
                save_future = await self.training_client.save_state_async(f"step_{step:06d}")
                saved_path = (await save_future.result_async()).path
                _log.info(f"Saved checkpoint at step {step} to {saved_path}")
                record_checkpoint(args.output_dir, saved_path, step)

            batch = load_training_batch(self.train_dataset, step, args.batch_size, args.seed)
            sampling_client = await self.training_client.save_weights_and_get_sampling_client_async()

            if step % args.eval_freq == 0:
                await self._evaluate(step, sampling_client)

            await self._train_step(step, batch, sampling_client)

        await self._finalize()

    async def _setup(self) -> None:
        """
        Set up the datasets, reward and label resolution, renderer and backends, logger, wandb, client.
        """
        args = self.args

        self.train_dataset = load_parquet_dataset(args.train_file)
        self.eval_datasets = {
            split: load_parquet_dataset(args[f"{split}_file"])
            if bool(args.get(f"do_{split}", True)) and args.get(f"{split}_file") else []
            for split in ("dev", "test")}
        print(f"Train: {len(self.train_dataset)}  Dev: {len(self.eval_datasets['dev'])}  "
              f"Test: {len(self.eval_datasets['test'])}  "
              f"(evaluating: {[s for s, d in self.eval_datasets.items() if d] or 'nothing'})")

        get_reward(args.true_reward)
        get_reward(args.misaligned_reward)
        get_hack_label(args.hack_label)
        optimized = optimized_reward_name(args.model_role, args.true_reward, args.misaligned_reward)
        _log.info(
            f"model_role={args.model_role}: optimizing {optimized!r} "
            f"(true_reward={args.true_reward!r}, misaligned_reward={args.misaligned_reward!r}, "
            f"hack_label={args.hack_label!r})"
        )
        self.intent_in_input = args.intent_in_input
        _log.info(f"intent_in_input={self.intent_in_input!r}: appending the task's intent to every policy input")

        self.description: str | None = get_description(args)
        if self.description:
            _log.info(f"Description for the conditioned half of every training group: {self.description!r}")


        self.tokenizer, self.renderer, self.stop_sequences, self.monitor_backend = \
            setup_generation_and_backends(args)

        self.logger = RunLogger(args.output_dir, n_traces_per_step=args.n_traces_train)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            cfg_path = os.path.join(args.output_dir, "config.yaml")
            if not os.path.exists(cfg_path):
                OmegaConf.save(config=args, f=cfg_path)

        self.use_wandb = bool(args.get("use_wandb", False))
        if self.use_wandb:
            import wandb
            if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
                raise RuntimeError("use_wandb=true but no wandb credentials: set WANDB_API_KEY in keys.sh, "
                                   "or pass use_wandb=false.")
            wandb.init(
                entity=args.get("wandb_entity"),
                project=args.get("wandb_project"),
                name=args.get("wandb_run_name") or (Path(args.output_dir).name if args.output_dir else None),
                config=OmegaConf.to_container(args, resolve=True),
                dir=args.output_dir,
            )
            _log.info(f"Initialized wandb run: {wandb.run}")

        service_client = tinker.ServiceClient()
        self.training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model, rank=args.lora_rank
        )
        self.adam_params = tinker.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.95)

    async def _evaluate(self, step: int, sampling_client) -> None:
        """
        Evaluate every enabled split, skipping a split whose dump for this step already exists.
        """
        args = self.args
        out = args.output_dir
        for split, dataset in self.eval_datasets.items():
            if not dataset:
                continue
            dump = os.path.join(out, f"inference_{split}", f"step_{step:06d}.jsonl") if out else None
            if dump and os.path.exists(dump):
                _log.info(f"inference_{split}/step_{step:06d}.jsonl exists; skipping {split} evaluation.")
                continue
            await run_evaluation(
                step=step, eval_dataset=dataset, sampling_client=sampling_client,
                renderer=self.renderer, tokenizer=self.tokenizer, args=args, stop_sequences=self.stop_sequences,
                monitor_backend=self.monitor_backend, logger=self.logger,
                eval_output_dir=os.path.join(out, f"inference_{split}") if out else None,
                split_name=split,
            )

    async def _resume(self) -> int:
        """
        Reload weights, optimizer state and the step counter; returns start_step.
        """
        args = self.args
        start_step = 0
        if bool(args.get("resume", True)):
            loaded_state = load_training_state(args.output_dir)
            if loaded_state is not None:
                start_step = int(loaded_state.get("step", 0))
                _log.info(f"Resuming from training state: step={start_step}")
            latest_txt = os.path.join(args.output_dir, LATEST_CHECKPOINT_FILENAME) if args.output_dir else None
            if latest_txt and os.path.exists(latest_txt) and loaded_state is None:
                raise RuntimeError(
                    f"{latest_txt} exists but training_state.json is missing or unreadable, so "
                    "the step counter for those weights is unknown. Resuming would restart from "
                    "step 0 on trained weights. Delete latest_checkpoint.txt to start fresh, or "
                    "restore training_state.json."
                )
            latest_ckpt_path = latest_checkpoint_path(args.output_dir) if args.output_dir else None
            if latest_ckpt_path:
                load_future = await self.training_client.load_state_with_optimizer_async(latest_ckpt_path)
                await load_future.result_async()
                _log.info(f"Resumed training client (weights+optimizer) from: {latest_ckpt_path}")
        return start_step


    def _plain_and_conditioned_inputs(self, row: dict) -> tuple[str, str | None]:
        """
        Build (plain input, plain input plus the description); the second is None without a description.
        """
        plain = build_policy_input(row, self.intent_in_input)
        if not self.description:
            return plain, None
        return plain, add_description_prompt(plain, self.description)

    async def _rollout_and_score(self, batch: list[dict], sampling_client):
        """
        Sample group_size rollouts per instance and score them. With a reward description, half of each
        group is sampled with it appended; the gradient is calculated conditioned on the prompt without the description.
        Returns (train_prompts, tokens_per_prompt, sequences_flat, response_texts_flat, flat_scores, train_logprobs, conditioned_per_prompt).
        """
        args = self.args
        train_prompts: list = []
        response_texts_flat: list[str] = []
        sequences_flat: list = []
        score_requests = []

        train_sampling_params = build_sampling_params(args, self.stop_sequences, prefix="train_")
        n_conditioned = args.group_size // 2 if self.description else 0
        n_plain = args.group_size - n_conditioned

        sample_requests, call_destinations = [], []
        for row_index, row in enumerate(batch):
            plain_input, conditioned_input = self._plain_and_conditioned_inputs(row)
            plain_prompt = self.renderer.build_generation_prompt([{"role": "user", "content": plain_input}])
            train_prompts.append(plain_prompt)
            if n_conditioned:
                conditioned_prompt = self.renderer.build_generation_prompt(
                    [{"role": "user", "content": conditioned_input}])
                sample_requests.append(sampling_client.sample_async(
                    prompt=conditioned_prompt, num_samples=n_conditioned, sampling_params=train_sampling_params))
                call_destinations.append((row_index, True, n_conditioned))
            sample_requests.append(sampling_client.sample_async(
                prompt=plain_prompt, num_samples=n_plain, sampling_params=train_sampling_params))
            call_destinations.append((row_index, False, n_plain))

        sequences_per_prompt: list[list] = [[] for _ in batch]
        conditioned_per_prompt: list[list[bool]] = [[] for _ in batch]
        for (row_index, conditioned, n), result in zip(
                call_destinations, await asyncio.gather(*sample_requests), strict=True):
            if len(result.sequences) != n:
                raise RuntimeError(
                    f"sampler returned {len(result.sequences)} sequences for num_samples={n}; "
                    "the per-instance group assembly assumes exactly that many.")
            sequences_per_prompt[row_index].extend(result.sequences)
            conditioned_per_prompt[row_index].extend([conditioned] * n)

        for row, seqs in zip(batch, sequences_per_prompt, strict=True):
            for seq in seqs:
                toks = list(seq.tokens)
                response_texts_flat.append(decode_full_text(toks, self.tokenizer))
                sequences_flat.append(seq)
                cot, final_output = parse_cot_final(
                    toks, self.tokenizer, args.think_start_str, args.think_end_str
                )
                score_requests.append(
                    score_response(
                        row,
                        model_role=args.model_role,
                        true_reward=args.true_reward,
                        misaligned_reward=args.misaligned_reward,
                        cot=cot,
                        final_output=final_output,
                        verifier_settings=args.get("verifier_settings"),
                    )
                )

        flat_scores = list(await asyncio.gather(*score_requests))

        max_token_id = args.max_token_id
        tokens_per_prompt = [[[t for t in seq.tokens if max_token_id is None or t <= max_token_id] for seq in seqs]
                             for seqs in sequences_per_prompt]
        do_recompute_per_prompt = [[conditioned or len(toks) < len(seq.tokens)
                                 for conditioned, toks, seq in zip(flags, group_tokens, seqs, strict=True)]
                                for flags, group_tokens, seqs in zip(conditioned_per_prompt, tokens_per_prompt,
                                                                     sequences_per_prompt, strict=True)]
        train_logprobs = await self._logprobs_under_training_prompt(
            train_prompts, tokens_per_prompt, sequences_per_prompt, do_recompute_per_prompt, sampling_client)
        return (train_prompts, tokens_per_prompt, sequences_flat, response_texts_flat,
                flat_scores, train_logprobs, conditioned_per_prompt)

    async def _logprobs_under_training_prompt(self, train_prompts: list,
                                              tokens_per_prompt: list[list[list[int]]],
                                              sequences_per_prompt: list[list],
                                              do_recompute_per_prompt: list[list[bool]], sampling_client
                                              ) -> list[list[list[float]]]:
        """
        Compute the logprobs of each flagged rollout's training tokens under the plain prompt at the
        current weights; unflagged rollouts keep their sampler logprobs.
        """
        out = [[list(seq.logprobs) for seq in seqs] for seqs in sequences_per_prompt]
        result_positions, logprob_requests = [], []
        for i, (prompt, group_tokens, flags) in enumerate(
                zip(train_prompts, tokens_per_prompt, do_recompute_per_prompt, strict=True)):
            for j, (tokens, recompute) in enumerate(zip(group_tokens, flags, strict=True)):
                if not recompute:
                    continue
                result_positions.append((i, j, len(tokens)))
                logprob_requests.append(sampling_client.compute_logprobs_async(
                    prompt.append(tinker.EncodedTextChunk(tokens=list(tokens)))))
        for (i, j, n_tokens), values in zip(result_positions,
                                            await asyncio.gather(*logprob_requests), strict=True):
            tail = list(values)[-n_tokens:]
            if any(v is None for v in tail):
                raise RuntimeError(
                    f"compute_logprobs returned None inside the {n_tokens}-token response; "
                    "importance_sampling needs a real logprob for every token.")
            out[i][j] = tail
        return out

    async def _train_step(self, step: int, batch: list[dict], sampling_client) -> None:
        """
        Run one GRPO step: roll out, score, build datums, update, log.
        """
        args = self.args
        train_prompts, tokens_per_prompt, sequences_flat, response_texts_flat, flat_scores, \
            train_logprobs, conditioned_per_prompt = await self._rollout_and_score(batch, sampling_client)

        g = args.group_size
        all_response_texts: list[list[str]] = [
            response_texts_flat[i * g: (i + 1) * g] for i in range(len(batch))
        ]
        all_score_results: list[list[dict]] = [
            flat_scores[i * g: (i + 1) * g] for i in range(len(batch))
        ]
        rewards_per_instance: list[list[float]] = [
            [s["reward"] for s in group] for group in all_score_results
        ]

        batch_metrics = compute_batch_metrics(flat_scores)


        datums, mean_rewards, n_identical_reward, n_harness_dropped = build_datums(
            train_prompts, tokens_per_prompt, train_logprobs, rewards_per_instance,
            usable_per_prompt=[[usable_for_training(s) for s in group] for group in all_score_results],
        )
        n_bad_tokens_stripped = sum(len(seq.tokens) - len(toks)
                                    for seq, toks in zip(sequences_flat, (t for g in tokens_per_prompt for t in g),
                                                         strict=True))
        n_bad_token_rollouts = sum(len(toks) < len(seq.tokens)
                                   for seq, toks in zip(sequences_flat, (t for g in tokens_per_prompt for t in g),
                                                        strict=True))
        if datums:
            fwd_bwd = await self.training_client.forward_backward_async(datums, loss_fn="importance_sampling")
            optim = await self.training_client.optim_step_async(self.adam_params)
            await fwd_bwd.result_async()
            await optim.result_async()

        self.logger.log_train_step(
            step=step,
            rows=batch,
            all_response_texts=all_response_texts,
            all_score_results=all_score_results,
            rewards_per_instance=rewards_per_instance,
            mean_rewards=mean_rewards,
            n_identical_reward=n_identical_reward,
            n_datums=len(datums),
            batch_metrics=batch_metrics,
        )

        rewards_flat = [r for grp in rewards_per_instance for r in grp]
        split_metrics: dict = {}
        split_metrics.update(compute_reward_stats(rewards_flat, rewards_per_instance))
        split_metrics.update(compute_generation_metrics(
            sequences_flat,
            [s.get("cot") for s in flat_scores],
            [s.get("final_output") for s in flat_scores],
            self.tokenizer,
            args.max_tokens,
        ))
        split_metrics.update(batch_metrics)
        step_metrics = {f"train/{k}": v for k, v in split_metrics.items()}
        step_metrics.update({
            "train/n_datums": len(datums),
            "train/n_harness_dropped": n_harness_dropped,
            "train/n_bad_token_rollouts": n_bad_token_rollouts,
            "train/n_bad_tokens_stripped": n_bad_tokens_stripped,
            "train/frac_identical_reward": (n_identical_reward / len(mean_rewards)) if mean_rewards else 0.0,
            "progress/step": step,
            "progress/done_frac": (step + 1) / args.n_steps,
            "optim/lr": args.learning_rate,
        })
        log_metrics(step_metrics, step=step, output_dir=args.output_dir, use_wandb=self.use_wandb)

    def _final_evaluation_done(self) -> bool:
        """
        Check whether every dump the final evaluation would write already exists (False without
        an output_dir, where nothing is dumped).
        """
        args = self.args
        if not args.output_dir:
            return False
        expected = [os.path.join(args.output_dir, f"inference_{split}", f"step_{args.n_steps:06d}.jsonl")
                    for split, dataset in self.eval_datasets.items() if dataset]
        return bool(expected) and all(os.path.exists(p) for p in expected)

    async def _finalize(self) -> None:
        """
        Evaluate the final weights, then save the final checkpoint and sampler weights.
        """
        args = self.args
        if self._final_evaluation_done():
            _log.info(f"Final evaluation dumps for step {args.n_steps} already exist; skipping.")
        else:
            final_sampling_client = await self.training_client.save_weights_and_get_sampling_client_async()
            await self._evaluate(args.n_steps, final_sampling_client)

        if args.output_dir:
            final_save = await self.training_client.save_state_async(f"step_{args.n_steps:06d}")
            final_ckpt_path = (await final_save.result_async()).path
            _log.info(f"Saved final checkpoint to {final_ckpt_path}")
            record_checkpoint(args.output_dir, final_ckpt_path, args.n_steps)

        save_future = await self.training_client.save_weights_for_sampler_async(f"step_{args.n_steps:06d}")
        save_result = await save_future.result_async()
        print(f"Saved model: {save_result.path}")

        if self.use_wandb:
            import wandb
            wandb.finish()

        self.logger.close()
