# RoboDojo 双臂 Spatial Memory

分支 `robodojo-spatial-mem` 从 `feature/0.8b-spatial-robodojo` 的
`3db1713` 切出。参考 `origin/libero-0.8B-spatial-mem` 的 execution memory、
learned slots、gated retrieval 和 waypoint conditioning，保留当前 RoboDojo
双臂空间预测头、50 步动作头及三相机输入。

## 数据与因果约束

- state/action 均为原始 14D absolute qpos，顺序为左臂 6 关节、左夹爪、
  右臂 6 关节、右夹爪。历史数据不做 q99；当前 state 和输出动作使用 checkpoint
  保存的训练 q99 transform，在线端复用同一实现。parquet 中的 float64 state/action
  在 q99 前统一为 float32，与在线输入精度一致，避免打包后才转换造成数值偏差。
- 当前图像顺序为 `cam_high, cam_left_wrist, cam_right_wrist`，分辨率 224×224。
  历史每路图像预处理为有序 2×4 个 8×8 RGB patch（`uint8`），训练和在线端共用
  `image_to_memory_tokens`，不重复解码模型视觉输入。
- 默认在当前环境步 `t` 采样 `t - 8 × [16, 15, ..., 1]`，负索引以零填充。
  离线 parquet 与在线反馈使用同一 `RoboDojoMemorySpec`。
- 一条记录为执行第 `s` 步之前的 raw state、第 `s` 步实际执行的 command，以及
  `t-s`。command 只有在该环境步完成后才可入库。未执行的预测 chunk 不可写入。
- 默认关闭历史 pose。开启时必须同时提供训练和在线的真实双臂 FK `[2,4,4]`，
  不使用单位矩阵占位。
- future pose、action labels、仿真器物体位姿、成功标志均不进入预测路径。
  future pose 只参与空间监督损失。在线服务器只向模型传递当前可观测输入与已执行历史。

## 架构与并行

29D 历史记录（14 state + 14 executed action + 1 elapsed）投影至 256D。
8 个 learned slots 的更新合并为一次批量 cuDNN GRU，避免按历史步循环 GRUCell。
右侧有效的历史先压紧再取最后有效输出；全 padding 时保留初始 slots。
action queries 用两层 attention 检索历史与 slots，经门控残差进入动作路径；
检索摘要同时调节 waypoint 路径。新增残差输出投影零初始化，可从 OFT 或 spatial
checkpoint 开始训练。

history 只在 DataFrame 中做一次 gather，不加载历史视频；DataLoader 开启
persistent workers、pin memory 和预取。训练使用 SDPA、自动 BF16（支持的 GPU）
及 fused AdamW；历史窗口和显存占用有界。

## 原始数据与额外数据分开

Memory 预处理只生成索引和视觉 patch，不复制原始 parquet、视频或统计数据。
原始数据 adapter 与新增数据应保存在同一目录树中：

```text
robodojo_spatial_training_bundle/
├── data/starvla_robodojo_v30/RoboDojo_lerobot_v30_video/
└── memory_sidecars/robodojo_v30_r16_s8_visual_v2/
    ├── manifest.json
    ├── source_dataset -> ../../data/starvla_robodojo_v30/RoboDojo_lerobot_v30_video
    └── episodes/
        ├── 000000.npy                # int32 [episode_length, 16]
        ├── 000000.visual.npy         # uint8 [episode_length, 3, 8, 8, 8, 3]
        └── ...
```

`source_dataset`、manifest 中的 source 路径，以及数据 adapter 下指向原始数据的链接
都使用相对路径。把 sidecar 上传到训练 bundle 的 `memory_sidecars/` 并保留版本目录，
链接会指向服务器已有的原始数据 adapter；
加载器还会校验 metadata 和 parquet SHA256，源数据不匹配时拒绝训练。当前机器
生成目录为 `/home/data/robodojo_cache/robodojo_spatial_training_bundle_20261013/memory_sidecars/robodojo_v30_r16_s8_visual_v2`。

索引 `.npy` 只保存严格过去的帧索引；无效位置为 -1；visual `.npy` 保存每帧三路
patch。训练加载器按 episode 和
frame index 从原始 parquet 一次 gather 原始 14D state/action，不复制视频、
state/action 或归一化统计，也不缓存未来标签。索引总量约 114 MiB；空间监督继续
使用已有独立 `spatial_sidecar_dir`。默认 `use_pose=false`，没有生成额外 pose。

manifest 记录原始路径、源 parquet 和元数据 SHA256、相机和双臂通道顺序、
memory 参数及每个 episode 的源 shard/起止行与索引文件校验值。构建按源 shard
并行，只读取数值列，拒绝帧缺失、错误维度、非有限数、通道错位和 episode 分片；
完整构建后原子发布目录，不覆盖已有版本。原始数据树也可通过 symlink 挂载。

仓库根目录、安装 pandas/pyarrow/numpy 的 Python 环境中执行（路径按实际部署设置）：

```bash
export ROBODOJO_SPATIAL_ROOT=/home/data/robodojo_cache/robodojo_spatial_training_bundle_20261013
PYTHONPATH=. python examples/simBenchmarks/RoboDojo/train_files/prepare_memory_sidecar.py \
  --data-root "$ROBODOJO_SPATIAL_ROOT/data/starvla_robodojo_v30/RoboDojo_lerobot_v30_video" \
  --sidecar "$ROBODOJO_SPATIAL_ROOT/memory_sidecars/robodojo_v30_r16_s8_visual_v2" \
  --workers 3
```

若目标目录已经存在，复核时在上述命令后添加 `--validate`，不会重新生成或写入数据。
变更窗口/步距时使用新的输出版本，并同步设置 `--recent-records`、
`--decision-stride` 和训练配置；开启 pose 时还需 `--use-pose`。

训练配置 `datasets.vla_data.memory_sidecar_dir` 已指向生成目录。加载器验证
source symlink、元数据、memory 参数、源文件与严格过去的索引网格，按需 mmap
当前 episode 的索引；每个 worker 对源 shard 只校验一次。源数据在训练期间应保持
只读。未配置该目录时保留原有按需计算索引的路径，两条路径使用相同因果约束。
在线推理仅使用已完成执行反馈构造相同网格，不需要访问离线目录或原始数据集。

## 训练与 A800 测速

配置：
`examples/simBenchmarks/RoboDojo/train_files/starvla_robodojo_v30_qwen35_0p8b_spatial_mem.yaml`。
默认单卡 batch=16、梯度累积=1。Stage A 冻结 Qwen 训练 5k 步；Stage B 从 A 的
final checkpoint 解冻全模型训练 45k 步。启动器先全量验证 memory 和已有空间
sidecar，通过后才启动训练，不会自动生成或覆盖数据：

```bash
python examples/simBenchmarks/RoboDojo/train_files/run_train_robodojo_spatial_mem.py
```

可设置 `PYTHON_BIN`、`CONFIG_YAML`、`ROBODOJO_SPATIAL_ROOT`；`STAGE_A_OVERRIDES` /
`STAGE_B_OVERRIDES` 接收空格分隔的 `key=value`，支持 shell 引号。Stage B 继承
Stage A 配置，并覆盖冻结策略、步数及初始化 checkpoint；默认 run_id 为
`robodojo_v30_qwen35_0p8b_spatial_mem_stage_a_full`。

在配置好 CUDA 动态库的训练环境、仓库根目录执行以下命令，分别测量两阶段：

```bash
PYTHONPATH=. python examples/simBenchmarks/RoboDojo/train_files/benchmark_spatial_mem.py \
  --stage a --require-fast-kernels --warmup 10 --steps 50 \
  --output playground/benchmarks/a800_mem_a.json
PYTHONPATH=. python examples/simBenchmarks/RoboDojo/train_files/benchmark_spatial_mem.py \
  --stage b --require-fast-kernels --warmup 10 --steps 50 \
  --output playground/benchmarks/a800_mem_b.json
```

Qwen3.5 的加速路径需要与 Torch/CUDA 兼容的 `flash-linear-attention` 和
`causal-conv1d`。`--require-fast-kernels` 会拒绝串行 PyTorch delta-rule 回退。
确认显存允许后，可增加 `--set trainer.gradient_checkpointing=false` 比较速度。
基准含真实数据读取/预处理、前后向、梯度裁剪和 optimizer update；排除启动和保存。
报告 batch、精度、显存、均值/中位数/p95、实际更新数；非有限损失或跳过更新不能算达标。

当前机器为 V100，无法验证 A800 的 1 秒/批目标。V100、FP16、batch=1、无上述
加速内核的短冒烟测量约为冻结阶段 1.74 秒/批、全量阶段 2.18 秒/批。
两阶段均预热 10 批、计时 3 批，各完成 3 次参数更新，损失有限、无跳过更新；
峰值显存分别为 3.67 / 12.99 GiB。短测量不能作为 A800 吞吐保证，
正式验收必须使用目标卡和实际 batch=16。

## 在线推理

`PolicyServerWrapper` 根据 checkpoint 的 `QwenSpatialMemOFT` 自动启用按
`(unnorm_key, env_id)` 隔离的历史。WebSocket 的每个 example 除 image/lang 外要求：

```python
example = {
    "image": [cam_high, cam_left_wrist, cam_right_wrist],
    "lang": instruction,
    "env_id": "env-0", "episode_id": "episode-001", "env_step": 0,
    "raw_state": raw_qpos_14, "memory_feedback": [],
}
# 按正常 predict_action 协议提交 examples=[example]；返回的是反归一化动作。
```

新 episode 必须从 env_step=0 和空 feedback 开始。假设执行 chunk 中前 16 步，
下一次请求 `env_step=16` 必须携带全部 16 条完成反馈：

```python
example["env_step"] = 16
example["raw_state"] = current_raw_qpos_14
example["memory_feedback"] = [
    {"source_step": s, "state": raw_state_before_step[s],
     "executed_action": actual_command_sent_to_robot[s],
     "images": [cam_high[s], cam_left_wrist[s], cam_right_wrist[s]]}
    for s in range(16)
]
```

`executed_action` 为裁剪、安全处理、融合后的实际 14D absolute qpos command，
不是尚未执行的模型输出，也不是执行后的测量 state。反馈步号连续、不重复，
覆盖上次成功推理以来所有已完成步骤。服务端在预测与反归一化成功后才提交历史；
失败请求可用原反馈重试。新 episode_id 自动重置对应环境；`reset_memory()` 清空全部。

GR00T ZMQ 接口将上述 env_id、episode_id、env_step、memory_feedback 放入
`get_action.options.execution_memory`；当前 raw state 从 `observation.state`
按 DataConfig 顺序提取，视频字典须按 high/left/right 顺序构造。
`get_modality_config` 会返回 memory 协议。GR00T `reset` 清空历史。
旧的无反馈 RoboDojo 客户端需要接入此协议后才能使用 memory checkpoint；
服务端会拒绝缺失反馈的请求，避免静默产生训练/推理偏差。

## 回归验证

`tests/test_robodojo_memory_contract.py` 覆盖逐帧离线/在线一致性、padding/滚动/重置、
16 步执行反馈、未来数据隔离、错误协议拒绝、GRU 数值和梯度等价、q99 一致性及
CUDA autocast/低精度参数。GR00T 协议测试覆盖执行反馈转发与标签过滤。

预处理测试还覆盖原始文件不变、symlink 挂载、校验值变化、行关联错误、当前/未来
索引注入拒绝，以及实际 memory 加载器与在线 buffer 的逐帧一致性。

本次 69 项回归测试全部通过。真实数据验证覆盖每个源 shard 的首尾 episode，
共 6 个 episode、4,286 帧，历史记录逐帧与在线 buffer 完全相等；6 个三相机完整
样本的当前 state 与 checkpoint 统计驱动的在线 q99 处理完全相等。真实 DataLoader
以 4 workers、batch=16 完成加载；所有源 parquet 和元数据的验证前后 SHA256
保持不变，全部 3,500 个索引文件通过校验。结果记录在
[`robodojo_memory_preprocessing_result.json`](robodojo_memory_preprocessing_result.json)。
本次仅执行预处理与加载验证，未启动长时间训练。
