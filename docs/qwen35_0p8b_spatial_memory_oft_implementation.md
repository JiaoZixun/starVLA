# Qwen3.5-0.8B Spatial-Memory OFT

状态：代码和数据 sidecar 已完成。今天的训练日志显示 Stage A 曾在 CUDA0 运行到 3,970/5,000 步后停止；没有 5,000 步 checkpoint，当前也没有活动训练进程，因此该阶段明天需要从 spatial checkpoint 重新运行。尚未完成独立的 NVIDIA EGL renderer 验证。

当前分支从 `spatial` 切出为 `libero-0.8B-spatial-mem`。模型保留 Qwen3.5-0.8B 和原 OFT action head，单次 Qwen 前向得到当前视觉语言 hidden states 和 action queries。新增的 memory encoder 不重新调用 Qwen，因此因果历史不会把 Qwen 前向次数扩大为 8 倍。

## 数据流和模块

当前 RGB、机器人状态和任务文本进入 Qwen；原 action queries 同时进入因果执行记忆和 waypoint 分支。记忆检索结果先通过每个 action query 的通道门控注入动作 queries，也通过零初始化的投影注入 waypoint decoder。waypoint decoder 根据当前 TCP 锚点预测相对路点 `[1, 4, 8, 16]`，再由独立的 spatial gate 将预测路点 cross-attend 回动作 queries，最后仍由原 OFT action head 输出 8 步动作。

memory 配置为宽度 256、4 个 heads、2 层、32 条近期记录和 8 个长期 slots。长期 slots 按时间顺序递归更新；样本中最近窗口之前的 episode 历史作为无梯度 burn-in，最近窗口每 8 条记录截断梯度。memory gate 和 spatial gate 分开记录，均为每个 action query 的通道门控。新残差输出投影零初始化，接入时保留 spatial checkpoint 的动作路径。

训练目标保留 masked L1 BC、有效未来路点的平移 Huber 和旋转损失、未来夹爪状态回归。effect supervision 暂时关闭（`effect_weight: 0.0`），因为 action-row 与观测的 pre/post-action phase 仍未完成核验；不使用未经确认的效果标签或预测残差。可选的 BC-priority 梯度投影在训练器中保留为消融，并记录 BC/aux 梯度范数、余弦和冲突标志。

## 数据和因果边界

原始 LeRobot Parquet 和视频保持只读。几何 pose sidecar 位于 `playground/DerivedData/libero_spatial_oft_v1`，memory RGB patch sidecar 位于 `playground/DerivedData/libero_mem_oft_v1`。memory sidecar 只保存每视角 8 个 8x8 RGB patch、episode/frame 映射和完整性 manifest，不复制原始数据。四套数据共 273,465 帧、1,693 episodes，20 Hz：

| 数据集 | 帧数 | episodes |
| --- | ---: | ---: |
| `libero_spatial_no_noops_1.0.0_lerobot` | 52,970 | 432 |
| `libero_object_no_noops_1.0.0_lerobot` | 66,984 | 454 |
| `libero_goal_no_noops_1.0.0_lerobot` | 52,042 | 428 |
| `libero_10_no_noops_1.0.0_lerobot` | 101,469 | 379 |

四个 memory sidecar 已在 `fastvla-fla` 环境中通过 `prepare_memory_sidecars.py --verify`。goal 数据有 1 个 wrist 视频的 PyAV 解码回退到系统 ffmpeg；源 SHA256 和 129 帧数均严格核验，未生成视频副本。manifest 将该事件记录为 `decoder_fallbacks`。

输入只包含当前及过去的图像、状态、已经执行的动作和已观察效果。历史 source frame 必须严格小于当前 frame，episode 起点清空 memory。未来 pose、未来 gripper、`future_valid` 和未执行的 action chunk 只用于监督或有效性掩码，绝不进入模型输入。评估器只在每个新 8 步 chunk 的第一个环境步提交实际发送给环境的动作；排队的剩余动作不会写入执行记忆。

训练 loader 设置 `memory_decision_frame_only: true` 和 `memory_decision_stride_env_steps: 8`，只从 stride 对齐的完整 action chunk 起点抽样，并设置 `drop_incomplete_action_chunks: true`，避免 episode 尾部 padding 进入 BC。当前 loader 是随机抽取单个决策点并附带严格因果历史；它不是把 8 个决策点拼成一个 batch，也不会增加 Qwen 前向次数。burn-in 在 memory encoder 内按 episode 顺序运行并禁用梯度。

## 训练配置

启动脚本：`examples/simBenchmarks/LIBERO/train_files/run_qwen35_0p8b_spatial_mem_oft_1gpu.sh`。阶段 A 从已训练 spatial checkpoint 加载 Qwen、action、waypoint 和 adapter，冻结这些模块并运行 5,000 步 memory/spatial warmup；阶段 B 从阶段 A checkpoint 恢复并全量训练到累计 50,000 步。两阶段均为 CUDA 0、batch size 32、8 个 DataLoader workers、persistent workers、pin memory 和 prefetch 2。脚本今天不执行 CUDA 验证；明天启动前再按仓库 `AGENTS.md` 完成 `nvidia-smi` 和 LIBERO-plus EGL renderer 检查。

```bash
cd /data1/jiaozx/starVLA
PYTHON=/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python \
GPU_ID=0 STAGE=all WANDB_MODE=disabled \
bash examples/simBenchmarks/LIBERO/train_files/run_qwen35_0p8b_spatial_mem_oft_1gpu.sh
```

本次 Stage A 日志测得约 0.66 秒/步（3,970 步约 43 分 35 秒），但该阶段冻结了 Qwen 和原 action path，不能直接代表全量训练速度。此前空间 OFT 全量训练约 1.1 秒/步；按 Stage A 约 55 分钟、Stage B 剩余 45,000 步约 13.8 小时估算，从头完成两阶段约 14.7 小时，建议预留 15–16 小时。明天完成独立 CUDA/EGL 检查后，以前 100–500 个全量训练步重新确认 ETA。memory sidecar 预处理不占训练 GPU 时间。

## 验证清单

1. 在 `fastvla-fla` 环境执行四个 sidecar 的 `--verify`，确认源 revision、frame map、SHA256 和 fallback 帧数。
2. 执行 Python 编译检查和 `git diff --check`。
3. 明天启动长训练前设置单卡 `CUDA_VISIBLE_DEVICES=0`，完成 `nvidia-smi` 与 `check_nvidia_egl.py` 验证；训练脚本现在会在开始长跑前自动检查 NVIDIA EGL renderer。
4. 训练期间监控 BC、waypoint、gripper、memory/spatial gate 分布以及 BC/aux 梯度夹角；gate 长期接近零时不能仅凭总 loss 判断模块有效。
5. 训练结束后使用相同 LIBERO suite、seed 和 episode 协议评估，并与 spatial OFT checkpoint 对照；在完成评估前不宣称成功率提升。
