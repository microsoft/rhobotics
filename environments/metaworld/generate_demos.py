#!/usr/bin/env python3
"""Generate MetaWorld demonstrations with the scripted expert, as a LeRobot dataset.

MetaWorld ships a hand-written expert policy for every task, so the demonstrations this
example finetunes on are produced locally in a few minutes -- nothing to download.

    python environments/metaworld/generate_demos.py --task assembly-v3 --episodes 100

Writes to ``<root>/<repo_id>``, which the training config then points ``root_dir`` at.
Only successful episodes are kept; the scripted experts are good but not perfect (a few
MT50 tasks solve well under 100% of the time), and a failed demonstration is not a
demonstration.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from environments.metaworld.env import MT50_TASKS, MetaworldEnvConfig, MetaworldEnvWrapper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("generate_demos")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="assembly-v3")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--repo_id", default=None, help="default: metaworld/<task>")
    p.add_argument("--root", default=None, help="default: the LeRobot cache")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_timesteps", type=int, default=300)
    p.add_argument("--fps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    task_id = next(int(k) for k, v in MT50_TASKS.items() if v["env_name"] == args.task)
    spec = MT50_TASKS[task_id]
    repo_id = args.repo_id or f"metaworld/{args.task}"

    env = MetaworldEnvWrapper(
        MetaworldEnvConfig(task_id=task_id, resolution=args.resolution,
                           max_episode_steps=args.max_timesteps)
    )
    expert = env.make_expert()
    res = args.resolution

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        robot_type="metaworld",
        root=args.root,
        use_videos=False,  # single small camera; parquet-inline is simpler to inspect
        features={
            "observation.image": {"dtype": "image", "shape": (3, res, res),
                                  "names": ["channels", "height", "width"]},
            "observation.state": {"dtype": "float32", "shape": (4,),
                                  "names": ["hand_x", "hand_y", "hand_z", "gripper"]},
            "action": {"dtype": "float32", "shape": (4,),
                       "names": ["dx", "dy", "dz", "gripper"]},
        },
    )

    kept = attempted = 0
    while kept < args.episodes:
        obs, _ = env.reset(seed=args.seed + attempted)
        attempted += 1
        frames, success = [], False
        for _ in range(args.max_timesteps):
            action = np.asarray(expert.get_action(env._last_raw_obs), dtype=np.float32)
            frames.append({
                "observation.image": obs["observation.image"][0, 0].numpy(),
                "observation.state": obs["observation.state"][0, 0].numpy(),
                "action": np.clip(action[: env.action_dim], -1.0, 1.0),
                "task": spec["prompt"],
            })
            obs, _r, done, _t, _i = env.step(torch.from_numpy(action[: env.action_dim]))
            if bool(np.asarray(done).reshape(-1)[0]):
                success = True
                break

        if not success:
            continue
        for f in frames:
            ds.add_frame(f)
        ds.save_episode()
        kept += 1
        if kept % 10 == 0:
            logger.info(f"  {kept}/{args.episodes} episodes ({attempted} attempted)")

    env.close()
    logger.info(
        f"wrote {kept} episodes for {args.task} to {ds.root} "
        f"(expert success rate {kept / attempted:.0%})"
    )
    logger.info(f"point the training config's dataset.root_dir at: {ds.root}")


if __name__ == "__main__":
    main()
