# Finetune and online-adapt on MetaWorld

This walks through the full loop on a single MetaWorld task: generate demonstrations, finetune
the Rho base policy on them, then improve it further **without touching its weights** by
learning which sampling noise to feed its flow sampler.

The whole example runs on one GPU and needs no downloaded dataset — MetaWorld ships scripted
expert policies, so the demonstrations are produced locally in a few minutes.

## Why steer the noise instead of finetuning again

Rho's action head is a flow-matching sampler: it integrates an ODE from an initial noise tensor
to an action chunk. Which noise you start from decides which action you get, so a small network
that picks the noise can change the policy's behavior while every policy weight stays frozen.

That buys two things. The prior is preserved exactly — there is no forgetting, because nothing
is overwritten. And the result still ships as one checkpoint: the noise policy folds into the
same `.pt` and serves through the normal inference path.

## 0. Install

```bash
pip install -e .
pip install -r requirements-metaworld.txt
```

MetaWorld renders through MuJoCo, so set `MUJOCO_GL=egl` on a headless machine. Every script
below defaults it for you.

## 1. Generate demonstrations

```bash
python environments/metaworld/generate_demos.py --task assembly-v3 --episodes 100
```

`assembly-v3` ("pick up a nut and place it onto a peg") is a good demonstration task: hard
enough that a short finetune leaves real headroom, easy enough that the scripted expert solves
it reliably. The script keeps only successful episodes and prints where it wrote the dataset —
roughly seven minutes of CPU for 100 episodes.

Other tasks work by name; see `environments/metaworld/mt50_tasks.json` for all 50. Prefer one
where the scripted expert is reliable, since a failed demonstration is not a demonstration.

## 2. Finetune the base policy

Point `dataset.root_dir` at the path step 1 printed, then:

```bash
accelerate launch --mixed_precision bf16 rho/training/train_accelerate.py \
    --config_path environments/metaworld/configs/train_metaworld_rho.yaml \
    --dataset.root_dir=<path from step 1>
```

The config is deliberately short — 5k steps on ~100 demonstrations. The goal is a policy that
is clearly doing the task but clearly imperfect, because a saturated policy has nothing left
for the next stage to show. Expect roughly 0.4–0.5 success on `assembly-v3`.

Evaluate it:

```bash
MUJOCO_GL=egl python environments/metaworld/eval.py \
    --config_path=environments/metaworld/configs/eval_metaworld_rho.yaml \
    --pretrained_checkpoint=outputs/training_metaworld/checkpoints/checkpoint_step_0005000
```

Note `execution_horizon: 8` in the eval config. The policy is trained to emit a 16-step chunk
but hand over after 8 (`n_action_steps`), and executing all 16 open-loop measurably costs
success on MetaWorld — in our measurements, mean success across 13 tasks was **0.53 at
horizon 8 versus 0.40 at horizon 16**, with no training involved. Keep training and execution
horizons matched.

## 3. Online adaptation

```bash
MUJOCO_GL=egl python environments/metaworld/online_adapt.py \
    --checkpoint outputs/training_metaworld/checkpoints/checkpoint_step_0005000 \
    --task assembly-v3 --out outputs/online_metaworld
```

Each episode: the noise policy proposes the sampling noise, the frozen policy denoises it into
an action chunk, and the scripted expert takes over at a randomly drawn step. Every action
actually executed over a query period — expert or policy — is inverted back into noise space,
and those noise targets train the noise policy.

Two design points worth understanding, because both are easy to get wrong:

**The expert takes over at a random time, not on failure.** Intervening only after things go
wrong teaches recovery from bad states. Intervening at an arbitrary point teaches the policy to
avoid reaching them, which is what you want.

**The whole chunk is inverted, including the policy's own steps.** A chunk that straddles a
takeover contains policy actions then expert actions. Inverting only the expert tail and
padding the rest teaches the policy to freeze partway through a chunk.

### Reading the diagnostics

After the seed episodes the run prints two lines that answer different questions:

```
[inversion] round-trip mean=4.7e-06 max=3.1e-05 | dropped 0/45
[manifold]  |w| mean=0.21 p99=1.22 max=4.44 over_bound=0.001
```

`round-trip` is whether the inversion solved at all — decode the recovered noise and you should
get back the actions it came from. If this is large, the noise targets are not trustworthy and
those chunks are dropped rather than trained on.

`|w|` is whether the target is *reachable*. The noise policy emits `tanh(·) × magnitude`, so a
target beyond `magnitude` cannot be produced no matter how long you train. These fail
independently: an inversion can round-trip perfectly and still be unreachable. If the loss
plateaus early, check `max` and `over_bound` before touching the learning rate.

Raising `--magnitude` is the obvious response to an over-bound tail, and in our MetaWorld runs
it did not help — 3.0 beat both 2.0 and 8.0, and raising it to 8.0 failed to rescue tasks that
regressed even though the bound stopped binding. Treat the manifold line as a diagnostic, not
a knob to chase.

## 4. Fold and serve

Training writes `noise_head.pt`. Fold it into the policy:

```bash
python scripts/fold_noise_policy.py \
    --checkpoint <policy.pt> \
    --head outputs/online_metaworld/noise_head.pt \
    --out policy_with_sampler.pt
```

The result is one checkpoint carrying both. It evaluates exactly like any other — the flow
model finds the noise policy as a submodule and uses it instead of a Gaussian draw:

```bash
MUJOCO_GL=egl python environments/metaworld/eval.py \
    --config_path=environments/metaworld/configs/eval_metaworld_rho.yaml \
    --pretrained_checkpoint=policy_with_sampler.pt
```

Loading that checkpoint with `noise_policy` unset is legitimate and supported — it means "run
this the standard Gaussian way", and the folded tensors are dropped with a log line rather than
failing the load. That makes the before/after comparison a config change, not a second
checkpoint.

## What to expect

On the tasks where this helps, it helps substantially. Measured on a Rho policy finetuned
across MetaWorld-50, with 50 adaptation episodes and 100-episode evaluations:

| task | base | after adaptation |
|---|---|---|
| `assembly` | 0.45 | 0.78 |
| `stick-pull` | 0.58 | 0.78 |
| `pick-out-of-hole` | 0.22 | 0.37 |
| `hand-insert` | 0.73 | 0.86 |

It does not help everywhere. Across 13 tasks the mean gain was **+0.06** at a 50-episode
budget, 7 of 13 significant, and `lever-pull` got reliably *worse* (0.50 → 0.29). Two
regularities are worth knowing before you pick a task: a task whose scripted expert is itself
unreliable will not improve, and a task already near saturation has nothing to gain. Check the
base success rate first.

These numbers come from a different base checkpoint than this tutorial produces, so treat them
as the shape of the result, not a target to reproduce exactly.
