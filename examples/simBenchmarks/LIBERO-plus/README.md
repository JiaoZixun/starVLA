# 🚀 LIBERO-plus zero shot Evaluation

This document provides instructions for reproducing our **zero shot experimental results** with LIBERO-plus.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO-plus` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `LIBERO-plus` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---


## ⬇️ 0. Download Checkpoints

We use models trained exclusively on LIBERO to perform zero-shot evaluation on LIBERO-plus: [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/libero). Their corresponding results on LIBERO-plus are summarized in the table below.

### 📊 Experimental Results

| Model               | Camera | Robot | Language|Light|Background|Noise|Layout|Total|
|---------------------|-------|--------|---------|--------|------|-------|-------|-------|
| OpenVLA  | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT  | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| ABot-M0  | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| **VLAct** | 73.9 | 68.4 | 81.5 | 96.7 | 96.7 | 86.0 | 83.3 | 82.6 |
| **Qwen2.5-VL-FAST**   | 19.6 | 27.6 | 74.5 | 75.2 | 71.0 | 27.4 | 62.7 | 48.9 |
| **Qwen2.5-VL-GR00T**   | 32.9 | 50.8 | 86.3 | 96.2 | 85.7 | 62.0 | 73.6 | 66.4 |
| **Qwen2.5-VL-OFT**   | 34.4 | 63.7 | 82.1 | 86.9 | 88.8 | 53.3 | 74.6 | 67.2 |
| **Qwen3-VL-OFT**   | 47.0 | 60.1 | 87.0 | 96.3 | 95.3 | 73.1 | 79.2 | 75.0 |
| **Qwen3-VL-PI**   | 64.3 | 57.2 | 82.8 | 94.2 | 94.0 | 79.6 | 78.2 | 77.0 |



---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO-plus repository](https://github.com/sylvestf/LIBERO-plus) to install the base `LIBERO-plus` environment.  



Afterwards, inside the `LIBERO-plus` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4 mujoco==3.2.3
```

---

## 🚀 2. Evaluation Workflow

LIBERO-plus rendering must use the selected NVIDIA GPU through EGL. The launchers
reject non-EGL settings, verify the NVIDIA EGL vendor file, and create a test EGL
context before starting evaluation. Set `GPU_ID` to the GPU index visible to the
host (for example `GPU_ID=0`). Keep this rendering rule for future LIBERO-plus
evaluation and training scripts; other benchmark rendering configurations are
independent and should remain unchanged.

### Checkpoints and evaluation protocol

The local 0.8B checkpoints for zero-shot evaluation are:

| Model | Checkpoint |
|---|---|
| OFT baseline | `playground/Checkpoints/qwen35_0p8b_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt` |
| Spatial OFT | `playground/Checkpoints/qwen35_0p8b_spatial_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt` |

For each model, use the same checkpoint in `run_policy_server.sh` and
`eval_libero.sh`, restarting the server when switching checkpoints. The regular
launcher uses the StarVLA policy server on port 9883; the LIBERO-plus process
connects over the StarVLA client interface and reads that checkpoint's action
normalization statistics (`franka`). Spatial OFT also receives the current
end-effector pose from the simulator. The Nebula parallel evaluator instead
loads the checkpoint in its evaluation workers and runs inference in-process.

The full benchmark contains 10,030 tasks. The provided shell launchers run one
episode per task, so the full run contains 10,030 episodes:

| Suite | Tasks | Episodes per task |
|---|---:|---:|
| `libero_10` | 2,519 | 1 |
| `libero_goal` | 2,591 | 1 |
| `libero_object` | 2,518 | 1 |
| `libero_spatial` | 2,402 | 1 |
| **Total** | **10,030** | **1** |

These counts and the one-episode setting are defined by the parallel launcher's
suite slices and `num_trials_per_task=1`; direct Python evaluator defaults are
50 trials per task, so use the provided launchers for this protocol.

### Local Spatial OFT evaluation on one GPU

Use the `libero-0.8B-spatial` branch for the pure Spatial OFT checkpoint above.
The `libero-0.8B-spatial-mem` branch changes the shared spatial architecture
(including the spatial projection from 11 to 12 input features) and cannot
strictly load this checkpoint. The local launcher checks strict checkpoint
loading before starting workers; keep strict loading enabled.

From the repository root, run:

```bash
MODEL_SELECTION=spatial EVAL_GPU_ID=0 TASKS_PER_GPU=2 SAVE_VIDEOS=0 \
  bash examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_local_dual_libero_plus.sh
```

`EVAL_GPU_ID` is the physical GPU index for both CUDA inference and NVIDIA EGL
rendering. The launcher validates it with `nvidia-smi`, exports `MUJOCO_GL=egl`
and `PYOPENGL_PLATFORM=egl`, pins `__EGL_VENDOR_LIBRARY_FILENAMES` to the NVIDIA
GLVND JSON, and checks the selected `MUJOCO_EGL_DEVICE_ID` using
`examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py` before evaluation.
Add `PREFLIGHT_ONLY=1` to run checks without launching evaluation.

The four suites run concurrently on the selected GPU. With `TASKS_PER_GPU=2`,
there are eight workers in total, using the same midpoint slices as the local
baseline evaluation. Keep this partition fixed when comparing results: sensor
noise consumes a process-local random stream. `MODEL_SELECTION=both` evaluates
the baseline and spatial models sequentially on the selected GPU. Results are
aggregated automatically into `spatial/overall_results.json` under `RESULT_ROOT`.
An earlier complete run on one A100 took about 27 hours; sensor-noise tasks
dominate the tail, so standard LIBERO timings do not predict this duration.

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **starVLA environment**: runs the inference server.  
- **LIBERO-plus environment**: runs the simulation.  

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
bash examples/simBenchmarks/LIBERO-plus/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/simBenchmarks/LIBERO-plus/eval_files/run_policy_server.sh`  


---

### Step 2. Start the simulation (LIBERO-plus environment)

In the second terminal, activate the `LIBERO-plus` conda environment and run:  

```bash
bash examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.


---

⚠️ **Note:** Since LIBERO-plus has 10,030 tasks, completing all the evaluations will take an extremely long time. It is recommended to run multiple model instances in parallel for the evaluations. We provide code and scripts for parallel testing on cluster `./parallel_eval/run_nebula_libero_plus`. Please modify them to fit your own cluster.

