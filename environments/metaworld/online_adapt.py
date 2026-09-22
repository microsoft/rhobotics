#!/usr/bin/env python3
"""Online adaptation on MetaWorld: improve a finetuned policy without touching its weights.

The finetuned policy stays frozen. A noise policy learns to pick the sampling noise, trained
on expert corrections inverted back into noise space (see ``rho.online``). MetaWorld's
scripted expert stands in for the human operator, so the whole loop runs unattended.

    MUJOCO_GL=egl python environments/metaworld/online_adapt.py \
        --checkpoint outputs/training_metaworld/checkpoints/checkpoint_step_0005000 \
        --task assembly-v3 --out outputs/online_metaworld

The noise policy is a submodule of the flow model, so it trains in place and is written by
the normal checkpoint path. The result is an ordinary checkpoint that evaluates like any
other -- there is nothing to fold in afterwards.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from environments.metaworld.env import MT50_TASKS, MetaworldEnvConfig, MetaworldEnvWrapper  # noqa: E402
from rho.eval.eval_config import load_configs_from_checkpoint  # noqa: E402
from rho.eval.policy_interface import PolicyInterface, PolicyInterfaceConfig  # noqa: E402
from rho.online import (  # noqa: E402
    AdaptationConfig,
    InterventionSchedule,
    NoiseTargetBuffer,
    OnlineAdapter,
    summarize_inversions,
)
from rho.policies import make_policy  # noqa: E402
from rho.checkpoints import save_checkpoint_bundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("online_adapt")


def run_episode(adapter, env, expert, schedule, seed, cfg, use_noise_policy):
    """One episode. Returns (samples, success, diagnostics).

    Every executed action over a query period is accumulated -- expert or policy -- and the
    whole chunk is inverted. A chunk counts as corrected if the expert held control for any
    part of it.
    """
    obs, _ = env.reset(seed=seed)
    if schedule is not None:
        schedule.reset()

    samples, diags = [], []
    chunk_obs = chunk_emb = actions = None
    executed: list[np.ndarray] = []
    corrected = False
    success = False
    t = 0

    def finalize():
        nonlocal executed, corrected
        if chunk_obs is None or not executed or not corrected:
            executed, corrected = [], False
            return
        ex = np.array(executed, dtype=np.float32)
        if len(ex) < adapter.chunk:  # pad to chunk_size, as training-time padding does
            ex = np.concatenate([ex, np.repeat(ex[-1:], adapter.chunk - len(ex), axis=0)])
        w, st = adapter.invert(chunk_obs, ex[: adapter.chunk])
        st["kept"] = st["rt"] <= cfg.inversion_mse_threshold
        diags.append(st)
        if st["kept"]:
            samples.append((chunk_emb, w.reshape(-1)))
        executed, corrected = [], False

    while t < cfg.max_timesteps:
        if t % cfg.query_freq == 0:
            finalize()
            chunk_obs = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                         for k, v in obs.items()}
            chunk_emb = adapter.embed_for_buffer(chunk_obs)
            if use_noise_policy:
                noise = adapter.propose_noise(chunk_obs)
            else:  # before the first update, serve the policy's own Gaussian draw
                noise = np.random.randn(adapter.chunk, adapter.noise_dim).astype(np.float32)
            actions = adapter.act(chunk_obs, noise)

        intervening = schedule.intervening_at(t) if schedule is not None else False
        if intervening:
            a = np.asarray(expert.get_action(env._last_raw_obs), dtype=np.float32)
            corrected = True
        else:
            a = np.asarray(actions[t % cfg.query_freq], dtype=np.float32)

        executed.append(a[: env.action_dim].copy())
        obs, _r, done, _tr, _i = env.step(torch.from_numpy(a[: env.action_dim]))
        t += 1
        if bool(np.asarray(done).reshape(-1)[0]):
            success = True
            break

    finalize()
    return samples, success, diags


def evaluate(adapter, env, episodes, cfg, use_noise_policy, seed0=10_000):
    wins = 0
    for e in range(episodes):
        _, ok, _ = run_episode(adapter, env, None, None, seed0 + e, cfg, use_noise_policy)
        wins += int(ok)
    return wins / max(episodes, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="assembly-v3")
    p.add_argument("--out", required=True)
    # 2000 BC steps = 20 adaptation episodes; with 10 seed episodes that is 30 rollouts.
    # Success peaks there on assembly-v3 and declines with further training.
    p.add_argument("--max_steps", type=int, default=2000, help="BC gradient steps")
    p.add_argument("--bc_steps_per_episode", type=int, default=100)
    p.add_argument("--bc_batch_size", type=int, default=256)
    p.add_argument("--bc_lr", type=float, default=1e-4)
    p.add_argument("--seed_expert_episodes", type=int, default=10)
    p.add_argument("--magnitude", type=float, default=3.0)
    p.add_argument("--query_freq", type=int, default=8)
    p.add_argument("--max_timesteps", type=int, default=200)
    p.add_argument("--beta_start", type=float, default=1.0)
    p.add_argument("--beta_end", type=float, default=0.1)
    p.add_argument("--beta_decay_episodes", type=int, default=2000)
    p.add_argument("--takeover_max", type=int, default=75)
    p.add_argument("--eval_episodes", type=int, default=30)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_id = next(int(k) for k, v in MT50_TASKS.items() if v["env_name"] == args.task)

    cfg = AdaptationConfig(
        bc_lr=args.bc_lr, bc_batch_size=args.bc_batch_size,
        bc_steps_per_episode=args.bc_steps_per_episode, max_steps=args.max_steps,
        seed_expert_episodes=args.seed_expert_episodes, query_freq=args.query_freq,
        max_timesteps=args.max_timesteps, eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval, seed=args.seed,
    )

    pcfg, dcfg = load_configs_from_checkpoint(args.checkpoint)
    # Ask for a noise policy the finetuned checkpoint does not have yet: the flow model builds
    # it as a submodule, load_state_dict initialises it, and it trains in place.
    pcfg.noise_policy = "vlm_direct"
    pcfg.noise_policy_kwargs = dict(
        emb_dim=pcfg.embed_dim,
        state_dim=4,  # MetaWorld: hand xyz + gripper
        magnitude=args.magnitude,
        noise_steps=pcfg.chunk_size,
        noise_dim=pcfg.max_action_dim,
    )
    policy = make_policy(pcfg)
    policy.load_from_pretrained(args.checkpoint)
    policy.eval()
    pi = PolicyInterface(PolicyInterfaceConfig(data_config=dcfg, policy=policy, device="cuda"))
    adapter = OnlineAdapter(pi, policy, cfg)
    logger.info(f"task {args.task} (id {task_id}) | chunk {adapter.chunk} "
                f"noise_dim {adapter.noise_dim} | magnitude {args.magnitude}")

    eval_env = MetaworldEnvWrapper(MetaworldEnvConfig(task_id=task_id,
                                                      max_episode_steps=args.max_timesteps))

    env = MetaworldEnvWrapper(MetaworldEnvConfig(task_id=task_id,
                                                 max_episode_steps=args.max_timesteps))
    expert = env.make_expert()
    schedule = InterventionSchedule(
        beta_start=args.beta_start, beta_end=args.beta_end,
        decay_episodes=args.beta_decay_episodes, takeover_max=args.takeover_max, seed=args.seed,
    )
    buf = NoiseTargetBuffer()
    log: list[dict] = []

    base_sr = evaluate(adapter, eval_env, args.eval_episodes, cfg, use_noise_policy=False)
    logger.info(f"[step 0] BASE success rate (Gaussian noise): {base_sr:.3f}")
    log.append({"step": 0, "sr": base_sr, "kind": "base"})

    # Seed with expert-only episodes so the first BC update has something to fit.
    schedule.force_immediate = True
    seed_diags = []
    for e in range(args.seed_expert_episodes):
        s, ok, d = run_episode(adapter, env, expert, schedule, args.seed + e, cfg, False)
        for emb, w in s:
            buf.add(emb, w)
        seed_diags += d
        logger.info(f"  seed ep {e + 1}/{args.seed_expert_episodes}: success={ok} "
                    f"chunks {len(s)}/{len(d)} kept  buffer={len(buf)}")
    schedule.force_immediate = False

    stats = adapter.calibrate_embedding_stats(buf)
    logger.info(f"[calibration] embedding stats from {stats['n']} chunks: "
                f"mean|mu|={stats['emb_mean_abs']:.2f} mean(sigma)={stats['emb_std_mean']:.2f}")

    if seed_diags:
        d = summarize_inversions(seed_diags)
        logger.info(f"[inversion] round-trip mean={d['rt_mean']:.2e} max={d['rt_max']:.2e} | "
                    f"dropped {d['dropped']}/{d['n']}")
        logger.info(f"[manifold]  |w| mean={d['w_mean']:.2f} p99={d['w_p99']:.2f} "
                    f"max={d['w_max']:.2f} over_bound={d['over_bound']:.3f}")
        if d["w_max"] > args.magnitude:
            logger.info("            targets exceed the noise policy's output range; if the BC "
                        "loss plateaus, that tail is why")
        json.dump(d, open(out / "seed_inversion.json", "w"), indent=1)

    step = episode = 0
    t0 = time.time()
    window: deque = deque(maxlen=400)
    while step < args.max_steps:
        s, ok, d = run_episode(adapter, env, expert, schedule,
                               args.seed + 1000 + episode, cfg, use_noise_policy=len(buf) > 0)
        for emb, w in s:
            buf.add(emb, w)
        window.extend(d)
        episode += 1
        if len(buf) == 0:
            continue

        losses = [adapter.bc_step(buf) for _ in range(args.bc_steps_per_episode)]
        step += args.bc_steps_per_episode

        if episode % 10 == 0:
            r = summarize_inversions(list(window))
            logger.info(f"step {step} ep {episode} buf {len(buf)} beta {schedule.beta:.2f} "
                        f"loss {np.mean(losses):.4f} ({(time.time() - t0) / 60:.0f}m)")
            if r:
                logger.info(f"    inversion: rt {r['rt_mean']:.2e} drop {r['dropped']}/{r['n']} "
                            f"| |w| p99 {r['w_p99']:.2f} max {r['w_max']:.2f}")

        if step % args.eval_interval < args.bc_steps_per_episode:
            sr = evaluate(adapter, eval_env, args.eval_episodes, cfg, use_noise_policy=True)
            logger.info(f"[step {step}] success rate: {sr:.3f}  (base {base_sr:.3f})")
            log.append({"step": step, "sr": sr, "kind": "online",
                        "loss": float(np.mean(losses)), "buffer": len(buf),
                        "inversion": summarize_inversions(list(window))})
            (out / "eval_log.json").write_text(json.dumps(log, indent=1))

    ckpt_dir = out / f"checkpoint_step_{step:07d}"
    save_checkpoint_bundle(policy, ckpt_dir, step=step, data_config=dcfg)
    logger.info(f"done: {episode} episodes, {step} BC steps, {(time.time() - t0) / 60:.1f} min")
    logger.info(f"wrote {ckpt_dir}")
    logger.info(f"evaluate with:  python environments/metaworld/eval.py "
                f"--config_path=environments/metaworld/configs/eval_metaworld_rho.yaml "
                f"--pretrained_checkpoint={ckpt_dir}")


if __name__ == "__main__":
    main()
