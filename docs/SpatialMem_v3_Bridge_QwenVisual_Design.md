# Spatial-Mem v3：Proprio/多层特征 Bridge + Qwen 视觉事件记忆

日期：2026-10-22　分支：`feature/libero-spatial-mem-v2`　代码提交：`bf1155b`

## 1. 背景：v2 为什么没有超过 Spatial OFT

v2（`qwen35_0p8b_spatial_mem_oft_v2_full_libero_50k`）以 Spatial OFT 50k 为初始化，继续训练 50k 步，同时加入了执行事件记忆。

| 指标 | Spatial OFT | Mem v2 |
|---|---:|---:|
| 标准 LIBERO 平均 | 91.85% | 91.50% |
| LIBERO-10 | 77.6% | 77.4% |
| LIBERO-plus（已完成的 6412 个任务，同一任务区间） | 38.86% | 37.87% |

诊断结论：memory 分支几乎没有被模型使用。

- 离线反事实测试（512 个训练样本）：action L1 在真实历史下为 0.02852，去掉历史后为 0.02858，差 0.2%；只保留最后 2 个事件时也是 0.02852。
- 同一批样本上 Spatial 50k 的 L1 是 0.0335。v2 比它低，几乎全部来自续训：`action_model` 的权重相对 Spatial 变化了 11.7%。
- `waypoint_twist_head` 训练结束时范数只有 0.033，说明基于历史的 waypoint 修正一直没有被打开。
- 事件的视觉输入是每个视角 4×4 RGB 均值，每个事件只有 96 个数，没有使用 Qwen 的视觉特征。
- 原因：LIBERO 示教接近马尔可夫，模仿学习的损失几乎不给 memory 分支梯度信号。

目标：完整模型在标准 LIBERO 上平均 ≥95%。Spatial、Object、Goal 已在 95–99%，因此 LIBERO-10 需要从 77.6% 提升到约 88% 以上。

## 2. v3 的修改内容

### 2.1 Proprio + 多层 VLM 特征 Bridge（对应之前的建议 1、2）

新增模块：`starVLA/model/framework/VLM4A/proprio_layer_bridge.py`

- **Query**：8 个 action token 在 Qwen 最后一层的特征。
- **Key/Value**：
  - Qwen 解码层 6/12/18/24 的完整序列，即 `hidden_states[i]`。每层先过各自的 LayerNorm，因为中间层的残差流没有经过归一化；再经过共享投影到 256 维，并加上 layer embedding。
  - token 类型 embedding 区分三类 token：文本、主视角图像、腕部视角图像。按每个样本中图像 token 的出现顺序二分，前半为主视角，后半为腕部视角。
  - 1 个本体感知 token，共 11 维：TCP 位置（3）+ TCP 旋转的 6D 表示（6）+ 夹爪 qpos（2）。旋转用 6D 而不用 axis-angle，因为 axis-angle 在接近 π 时会翻转符号。归一化用的 mean/std 是在 273,465 帧训练数据上统计的固定值，写在 config 里，训练和评估使用同一组数。
- **结构**：2 层 `WaypointDecoderBlock`（self-attn → cross-attn → FFN），宽度 256。
- **输出**：零初始化的 `Linear(256→1024)`，作为残差加到 action query 上。第 0 步时 bridge 输出为零，模型与 Spatial OFT 等价。
- **数据流**：bridge 处理后的 query 再进入 memory 残差、spatial adapter 和 action head。
- 这一步不新增推理时的输入：proprio 来自评估客户端已经发送的 `geometry_current.eef_pose` 和 `memory_state`，也不会写进 prompt。
- 多层特征通过 `Qwen3_5.forward_features(hidden_layers=...)` 在同一次前向里取出，不需要第二次调用 VLM。

### 2.2 事件视觉改用冻结 Qwen 视觉塔的特征（`memory.visual_source: qwen`）

- 每个事件的起点和终点视觉特征，改为 Qwen 视觉塔 post-merger token 在每个视角上 4×4 平均池化的结果，形状为 `[2, 16, 1024]`。
- 事件编码器对每帧先做 LayerNorm，再拼接 start、end 和 delta。对应 `ExecutionEventEncoder(visual_input_norm=True)`；RGB 模式下没有这一层，原有 checkpoint 的 state_dict 不变。
- **训练**：特征从离线 sidecar 读取，格式为 fp16。每个样本只传事件边界帧组成的 bank（相邻事件共用边界帧），在 GPU 上一次 gather 完成，避免在主机上为每个事件复制数据。
- **评估**：使用模型在同一 episode 之前几次 query 时自己算出的视觉特征，按 episode 和 step 缓存，只保留最近一次。客户端只发送事件元数据，即 `LiberoExecutionEventBuffer(include_visual=False)`；服务端通过 metadata 中的 `visual_source` 告诉客户端当前模式。

### 2.3 数据预处理（只保存额外数据）

脚本：`examples/simBenchmarks/LIBERO/train_files/prepare_qwen_visual_sidecar.py`

- 视觉权重来源：`qwen35_0p8b_spatial_oft_libero_50k/steps_50000`。v3 训练正是从这个 checkpoint reload `qwen_vl_interface`。
- 输出目录：`playground/DerivedData/libero_qwen_visual_v1/<dataset>/`
  - `obs_qwen_pooled.npy`：fp16 `[frames, 2, 16, 1024]`，四个 suite 共 273,465 帧，约 18 GB。
  - `manifest.json`：数据集 revision、帧映射（与 memv2 sidecar 一致）、相机顺序、预处理说明、视觉塔指纹。
- 源 Parquet 和 MP4 文件不复制、不修改。
- 帧的处理路径与训练一致：PyAV 解码，然后 PIL bicubic 缩放到 224×224。特征的处理器和池化代码与模型前向共用（`Qwen3_5.encode_pooled_images` 和 `_pool_image_features`）。

### 2.4 训练与评估一致性保证

| 环节 | 保证方式 | 验证结果 |
|---|---|---|
| 帧解码 | sidecar 的解码结果与训练 loader（`torchvision_av`）逐像素对比 | 两个视角最大差都是 0 |
| 视觉特征 | sidecar 与 batch-32 的模型前向对比 | 差为 0 |
| 视觉特征 | sidecar 与单样本在线前向（评估场景）对比 | 平均差 1.6e-4，特征量级约 0.09。原因是 bf16 batch 大小不同；原有 Qwen 主路径也有同样的差异 |
| 视觉塔权重 | `trainer.freeze_modules` 冻结；对 bf16 权重做 sha256 指纹，写入 `memory.visual_fingerprint`。如果视觉塔可训练、样本指纹不符，或模型自身权重指纹不符，训练和评估都会直接报错 | 指纹 `80fb666a…394f` |
| 事件字段 | `check_spatial_mem_v3.py` 用同一个 episode 对比训练路径和评估客户端路径 | state、动作、位姿的最大差为 0 |
| 动作输出 | 同一脚本把新增模块的零初始化输出层随机化后，对比训练路径与评估路径 | 最大差 0.0078（归一化单位）；作为对照，打开历史时动作变化 1.21 |
| 第 0 步等价 | 与 Spatial OFT 50k 对比动作 | 最大差 0.0039（bf16 舍入，动作平均量级 0.27） |

### 2.5 训练速度（要求 ≤1 s/step）

在 A100 上测 batch 32，前向 + 反向 + AdamW：

| 版本 | s/step | 峰值显存 |
|---|---:|---:|
| v2 | 0.898 | 26.5 GiB |
| v3 | 0.770 | 20.7 GiB |

v3 更快，因为冻结视觉塔后不再对它做反向传播。bridge 和 1024 维事件特征带来的开销小于这部分节省。单进程加载一个样本约 54 ms，8 个 worker 加载一个 batch 约 0.22 s，不会成为瓶颈。启动脚本会先把约 18 GB 的 sidecar 读进 page cache（机器内存 502 GB）。真实 trainer 下的速度会在正式训练开始前用一次短 smoke run 确认。

## 3. 配置与启动

- 配置：`examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_mem_oft_v3.yaml`
  - Spatial 50k 初始化（strict reload），训练 50k 步，batch 32，学习率与 v2 相同；`proprio_layer_bridge` 的学习率为 1e-4。
  - `freeze_modules: qwen_vl_interface.model.model.visual`
- 数据配比：mixture `libero_spatial_mem_oft_long2x`，Spatial:Object:Goal:Long = 1:1:1:2（采样概率，不按数据集长度缩放），LIBERO-10 占每个 batch 的 40%（原 `libero_spatial_mem_oft_all` 为各 25%）。
- 运行目录：`playground/Checkpoints/qwen35_0p8b_spatial_mem_oft_v3_long2x_libero_50k`

```bash
# 生成 sidecar（约 40 分钟，单卡）
python examples/simBenchmarks/LIBERO/train_files/prepare_qwen_visual_sidecar.py \
  --checkpoint playground/Checkpoints/qwen35_0p8b_spatial_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt
# 一致性检查
python examples/simBenchmarks/LIBERO/train_files/check_spatial_mem_v3.py --local-model-path <Qwen3.5 snapshot>
# 训练后跑标准 LIBERO；RUN_LIBERO_PLUS=1 时同时在另一张卡上跑 LIBERO-plus
TRAIN_STAGE=v3 RUN_LIBERO_PLUS=0 bash examples/simBenchmarks/LIBERO/train_files/run_spatial_mem_full_train_then_eval.sh
```

LIBERO-plus 仍按 AGENTS.md 的规则使用 NVIDIA EGL 渲染，并保持 `TASKS_PER_GPU=2` 的 worker 划分。

## 4. 风险与后续

- **缺少对照组。** v3 同样是"续训 50k 步 + 新模块"。要单独评估各个新模块的作用，需要两个对照：Spatial 不加新模块、以相同 lr 续训 50k 步；以及 v3 去掉 memory 只保留 bridge。
- **memory 仍可能学不起来。** v3 只增强了视觉输入，还没有加入必须依赖历史的辅助目标。训练日志中的 `memory_residual_ratio`、`bridge_residual_ratio` 和 `waypoint_twist_norm` 可以用来判断 memory 和 bridge 是否被使用。下一步候选：从夹爪开合周期推导子目标计数和阶段标签，作为辅助任务。
- **训练与评估的数值差。** 由 bf16 batch 大小引起，量级与原有 Qwen 主路径的差异相同，没有消除。
- **执行策略。** 是否改用 4 步重规划或时间集成，尚未决定。这属于评估协议的修改，需要单独确认。
