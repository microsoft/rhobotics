# Finetune and online-adapt on MetaWorld

This example runs the full loop on a single MetaWorld task: generate demonstrations, finetune
the Rho base policy on them, then improve it further **without changing any policy weight** by
learning which sampling noise to feed the flow sampler.

It runs on one GPU and needs no downloaded dataset — MetaWorld ships scripted expert policies,
so the demonstrations are produced locally.

## Why steer the noise

Rho's action head is a flow-matching sampler: it integrates an ODE from an initial noise tensor
to an action chunk. Which noise it starts from decides which action comes out, so a small
network that picks the noise can change the policy's behavior while the policy stays frozen.

The prior is preserved exactly, because nothing is overwritten. The noise policy is a submodule
of the flow model, so it trains in place and is saved with the model — the result is an ordinary
checkpoint that serves through the normal inference path.

## Install

```bash
pip install -e .
pip install -r requirements-metaworld.txt
```

MetaWorld renders through MuJoCo; the scripts below default `MUJOCO_GL=egl` so they work
headless.

If you have another `rho` package installed in the same environment, set `PYTHONPATH` to this
repository so the scripts import this copy:

```bash
export PYTHONPATH=$(pwd)
```

## 1. Generate demonstrations

```bash
python environments/metaworld/generate_demos.py --task assembly-v3 --episodes 100
```

`assembly-v3` is "pick up a nut and place it onto a peg". The script keeps only successful
episodes and prints the dataset path when it finishes. Takes a few minutes on CPU.

All 50 MT50 tasks are available by name; see `environments/metaworld/mt50_tasks.json`.

## 2. Finetune

Point `dataset.root_dir` at the path step 1 printed:

```bash
python rho/training/train_accelerate.py \
    --config_path=environments/metaworld/configs/train_metaworld_rho.yaml \
    --dataset.root_dir=<path from step 1> \
    --output_dir=outputs/training_metaworld
```

Roughly 20 minutes on one H100. The config trains for 1,000 steps at batch 32; the checkpoint
lands in `outputs/training_metaworld/<timestamp>/checkpoints/checkpoint_step_0001000`.

The short schedule is deliberate. It leaves the policy clearly capable but imperfect, which is
what the next step improves on; training to convergence saturates the task and there is nothing
left to show.

Evaluate it:

```bash
python environments/metaworld/eval.py \
    --config_path=environments/metaworld/configs/eval_metaworld_rho.yaml \
    --pretrained_checkpoint=<checkpoint path>
```

## 3. Online adaptation

```bash
python environments/metaworld/online_adapt.py \
    --checkpoint <checkpoint path> \
    --task assembly-v3 \
    --out outputs/online_metaworld
```

Each episode, the noise policy proposes the sampling noise, the frozen policy denoises it into
an action chunk, and the scripted expert takes over at a randomly drawn step. Every action
actually executed over a query period — expert or policy — is inverted back into noise space,
and those noise targets train the noise policy.

The run prints two diagnostic lines after the seed episodes:

```
[inversion] round-trip mean=2.7e-05 max=2.0e-04 | dropped 0/36
[manifold]  |w| mean=0.79 p99=7.57 max=12.38 over_bound=0.021
```

`round-trip` is whether the inversion solved: decode the recovered noise and you should get
back the actions it came from. Chunks above the threshold are dropped rather than trained on,
and `dropped` counts them.

`|w|` describes the noise targets themselves. The noise policy emits `tanh(·) × magnitude`, so
`over_bound` is the fraction of target values outside the range it can produce.

## 4. Evaluate the adapted policy

Adaptation writes a checkpoint when it finishes. The noise policy is part of the model, so it
evaluates exactly like the finetuned one:

```bash
python environments/metaworld/eval.py \
    --config_path=environments/metaworld/configs/eval_metaworld_rho.yaml \
    --pretrained_checkpoint=outputs/online_metaworld/checkpoint_step_0002000
```

To measure the same checkpoint without its noise policy — the before/after comparison — clear
the config field, and the flow model falls back to a Gaussian draw:

```bash
python environments/metaworld/eval.py \
    --config_path=environments/metaworld/configs/eval_metaworld_rho.yaml \
    --pretrained_checkpoint=outputs/online_metaworld/checkpoint_step_0002000 \
    --policy.noise_policy=null
```

The folded tensors are dropped with a log line rather than failing the load, so this is a
config change rather than a second checkpoint.

## Results

Measured end to end on `assembly-v3`. Each number is 30 evaluation episodes, pooled across
three adaptation runs:

| stage | success rate |
|---|---|
| base policy, 1,000 finetune steps | 0.53 |
| after online adaptation, 30 rollouts | **0.70** |

The adaptation used **30 environment rollouts** — 10 expert seed episodes and 20 adaptation
episodes, roughly five minutes of interaction. No policy weight changed; the entire difference
is which noise the sampler starts from.

Individual evaluations are noisy. Across 15 evaluations spanning three runs the success rate
ranged from 0.57 to 0.80, averaging 0.66. A single 30-episode number carries roughly +/-0.09 of
sampling noise, so treat one run as an estimate of a band, not a point.
