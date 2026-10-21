# 🚀 LIBERO Evaluation

This document provides instructions for reproducing our **experimental results** with LIBERO.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `LIBERO` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

> 💡 **AMD GPU Support:** Community members have verified that starVLA also works on **AMD Instinct MI300X** GPUs with ROCm 6.4 — with zero source code changes. The only modification needed is setting `--framework.qwenvl.attn_implementation sdpa`. For a detailed setup guide and benchmark results, see [Issue #254](https://github.com/starVLA/starVLA/issues/254).

---


## ⬇️ 0. Download Checkpoints


We provide a collection of pretrained checkpoints on Hugging Face to make community evaluation easier: [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/bench-libero). Their corresponding results on LIBERO are summarized in the table below.

### 📊 Experimental Results

| Model               | Steps | Epochs | Spatial | Object | Goal | Long  | Avg   |
|---------------------|-------|--------|---------|--------|------|-------|-------|
| $\pi_0$+FAST | -     | -      | 96.4    | 96.8   | 88.6 | 60.2  | 85.5  |
| OpenVLA-OFT | 175K  | 223    | 97.6    | 98.4   | 97.9 | 94.5  | 97.1  |
| $\pi_0$             | -     | -      | 96.8    | 98.8   | 95.8 | 85.2  | 94.1  |
| GR00T-N1.5 | 20K   | 203    | 92.0    | 92.0   | 86.0 | 76.0  | 86.5  |
| **VLAct** | - | - | 94.6 | 100.0 | 97.6 | 94.8 | 96.7 |
| **StarVLA-FAST (Qwen2.5-VL)** | 30K   | 9.54   | 97.3    | 97.2   | 96.1 | 90.2  | 95.2  |
| **StarVLA-OFT (Qwen2.5-VL)**  | 30K   | 9.54   | 97.4    | 98.0   | 96.8 | 92.0  | 96.1  |
| **StarVLA-π (Qwen2.5-VL)**    | 30K   | 9.54   | 98.2    | 99.2   | 95.6 | 88.4  | 95.4  |
| **StarVLA-GR00T (Qwen2.5-VL)**| 30K   | 9.54   | 97.8    | 98.2   | 94.6 | 90.8  | 95.4  |
| **StarVLA-FAST (Qwen3-VL)**   | 30K   | 9.54   | 97.3    | 97.4   | 96.3 | 90.6  | 95.4  |
| **StarVLA-OFT (Qwen3-VL)**    | 30K   | 9.54   | 97.8    | 98.6   | 96.2 | 93.8  | 96.6  |
| **StarVLA-π (Qwen3-VL)**      | 30K   | 9.54   | 98.8    | 99.6   | 95.8 | 88.4  | 95.7  |
| **StarVLA-GR00T (Qwen3-VL)**  | 30K   | 9.54   | 97.8    | 98.8   | 97.4 | 92.0  | 96.5  |

We train one policy for all 4 suites. All
scores are averaged over 500 trials for each task suite (10 tasks × 50 episodes).

---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base `LIBERO` environment.  

⚠️ **Common issue:** LIBERO defaults to Python 3.8, but the syntax updates between 3.8 and 3.10 are substantial. **We verified that using Python 3.10 avoids many issues**. 


Afterwards, inside the `LIBERO` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4 mujoco==3.2.3
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **starVLA environment**: runs the inference server.  
- **LIBERO environment**: runs the simulation.  

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
bash examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh`  

### Released Qwen3-PI LIBERO checkpoint compatibility

PR #373 introduced the canonical LayerwiseFM path used by current and newly
trained models. The released `StarVLA/Qwen3-VL-PI-LIBERO-4in1` checkpoint was
trained with historical LayerwiseFM semantics, so evaluate that checkpoint with
the compatibility override below:

```bash
CKPT=/path/to/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt \
USE_CANONICAL_FORWARD=false \
bash examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh
```

When `USE_CANONICAL_FORWARD` is unset, the server keeps the canonical default.
Do not edit the checkpoint `config.yaml`; use the launcher override instead.
The released checkpoint config uses 4 inference steps.


---

### Step 2. Start the simulation (LIBERO environment)

In the second terminal, activate the `LIBERO` conda environment and run:  

```bash
bash examples/simBenchmarks/LIBERO/eval_files/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.

Finally, each result will also save a video for visualization, as shown below:

![Example](example.gif)

---


# 🚀 LIBERO Training

## 📦 Step 0: Download the training dataset
Download the datasets to the playground/Datasets/LEROBOT_LIBERO_DATA directory:
- [LIBERO-spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

And move `modality.json` to each `$LEROBOT_LIBERO_DATA/subset/meta/modality.json`.

You could quickly prepare these by running:
```bash
# Set DEST to the directory where you want to store the data
export DEST=/path/to/your/data/directory
bash examples/simBenchmarks/LIBERO/data_preparation.sh
```


## 🚀 Step1: Start Training

Most of the required training files have been organized in [train_files](train_files).  

Please run the following command to start training:

```bash
bash examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh
```
⚠️ **Note:** Please ensure that you specify the correct path in `examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh`

### Qwen3.5 Spatial-Mem v2

The memv2 training configs use derived sidecars under
`playground/DerivedData/libero_spatial_mem_v2`. Generate them after downloading
all four LeRobot datasets:

```bash
python examples/simBenchmarks/LIBERO/train_files/prepare_memory_v2.py --no-source-hashes
bash examples/simBenchmarks/LIBERO/train_files/run_qwen35_0p8b_spatial_mem_oft_1gpu.sh
```

The preprocessing writes only episode/frame indexes, EEF-pose matrices and
two-camera 4×4 RGB summaries. It keeps source Parquet files and MP4 videos in
place. Set `PYTHON` and `GPU_ID` when using a specific environment or device.
Stage A loads the Spatial OFT checkpoint and trains only the new memory modules
for steps 1–5,000. Stage B resumes at step 5,000, unfreezes the full Qwen/OFT/
spatial/memory model and trains through total step 50,000 (45,000 more updates).
Both stages use batch size 32. `STAGE=a` and `STAGE=b` run the stages separately.
Before Stage B, the launcher checks for at least 33,000 MiB of free GPU memory;
the full-unfreeze batch peaks near 30 GiB and needs room for CUDA/Triton workspaces.
The v2 configs use a separate `qwen35_0p8b_spatial_mem_oft_v2_libero_50k`
run directory so existing v1 checkpoints remain intact. After training, start
standard LIBERO evaluation with
`examples/simBenchmarks/LIBERO/eval_files/run_qwen35_0p8b_spatial_mem_oft_after_training.sh`.

Each memory item is one completed 8-step interval. It contains the start/end
4×4 RGB summaries, the eight environment actions actually associated with the
interval, start/end proprioception, and episode-relative start/end EEF poses.
Training builds these intervals from the same stride-aligned LeRobot rows used
for the action targets. Evaluation closes an interval only after LIBERO accepts
its eight actions. A shared event validator enforces the shape, stride and
causal boundary. The event encoder compresses each interval into learned tokens;
a causal reader exposes only completed past intervals to action and waypoint
queries. Zero-initialized residual heads preserve the Spatial OFT behavior at
the start of memv2 training.

Train/eval contract notes:

- Memory proprioception is passed as `memory_state` (8-D, raw). `state` is never
  sent: Qwen OFT frameworks append `state` to the prompt, and the reused Spatial
  checkpoint was trained without it. QwenSpatialMemOFT rejects `state`.
- The Spatial 50k checkpoint predates the gripper head and spatial gate, so the
  memv2 configs set `spatial.adapter_version: v1` and `tau: [1, 2, 4, 8]`.
  Stage A uses `strict_reload: true`; a key mismatch stops training instead of
  freezing randomly initialized layers.
- Training samples every valid frame (`memory_decision_frame_only: false`). Each
  anchor uses complete 8-step events ending at the anchor frame. Evaluation
  always queries stride-aligned steps, with events starting at step 0.
- `memory.history_dropout` and `memory.event_action_dropout` act only in the
  training forward pass. Stage B re-warms the LR over `resume_warmup_steps`
  because Stage A does not save optimizer state.
- Both standard LIBERO and LIBERO-plus evaluators record every command accepted
  by `env.step`. LIBERO-plus must keep `TASKS_PER_GPU=2` to reproduce the
  baseline worker split and sensor-noise stream.
