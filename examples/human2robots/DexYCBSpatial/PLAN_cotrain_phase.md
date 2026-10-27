# 人手数据混合训练 + 阶段/物体监督 实施计划

分支：`libero-hand-cotrain-phase`（基于 `libero-hand-pretrain-spatial` @ `9770ca1`）
日期：2026-10-27

## 背景

DexYCB e20 两阶段预训练（先人手几何、再 LIBERO）在标准 LIBERO 上没有带来提升：
50K 时 90.70% vs Bridge-Spatial only 92.00%，Long 75.6% vs 81.0%；LIBERO 训练约 10K 步后两组损失曲线重合。
数据核查发现的对齐问题：

| 问题 | 现状 | 证据 |
|---|---|---|
| 域标识缺失 | `arm_embedding` / `arm_id` 对两个域都写死为 0 | `QwenSpatialOFT.py` WaypointModel.forward、SpatialAdapter.forward |
| anchor 依赖绝对坐标 | anchor 输入 = 绝对位置(3) + rot6d(6) | DexYCB 桌面系 z 0.08–0.50 m；LIBERO 机器人系 z 0.45–1.33 m，分布不重叠 |
| 提示词无任务语义 | 49,046 个样本全部是 "Predict the future motion of the end effector." | `samples.jsonl` |
| 速度不一致 | 同一秒数下人手位移约为 LIBERO 的 1.5–1.9 倍 | 中位 \|Δp\| (m)，0.05/0.1/0.2/0.4 s：人手 0.0121/0.0233/0.0434/0.0734，LIBERO 0.0064/0.0128/0.0255/0.0502 |
| 两阶段训练被冲掉 | 预训练只训 WaypointModel，之后 50K 步纯 LIBERO | 训练损失曲线 |

## 方案 1+2：混合训练 + 对齐修复

1. **按域区分 arm_id**：robot=0，human=1。`WaypointModel.forward` 新增 `arm_id` 参数（默认 0，旧调用方和旧 checkpoint 行为不变）。
   SpatialAdapter 只在 robot 域运行，保持 0。
2. **与坐标系无关的 anchor**：`spatial.anchor_mode`
   - `absolute`（默认，旧行为）：位置 + rot6d。
   - `gravity`：只用重力方向在末端局部坐标系中的表示 `R^T @ [0,0,-1]`（3 维，只反映末端倾斜，与世界原点和绕竖直轴的朝向无关），其余 6 维补零，保持 `anchor_projection` 输入为 9 维、参数形状不变。
     两个数据源的世界系都是 z 轴朝上，所以该量在两个域中含义一致。waypoint 标签本身是相对位姿，不需要绝对位置。
3. **时间重标定**：人手数据按 `time_scale` 采样未来帧，即 slot k 取 `FUTURE_SECONDS[k] / time_scale` 秒后的位姿，使人手的位移分布贴近 LIBERO。默认 `time_scale=1.7`（四个 tau 比值 1.89/1.82/1.70/1.46 的中间值），写入 sidecar metadata 并在生成后打印两域中位位移对比。
   选择重标定时间而不是缩放位移，是为了让旋转和位移保持一致。
4. **按序列的任务语言**：DexYCB `meta.yml` 的 `ycb_ids[ycb_grasp_ind]` 给出被抓物体，提示词为 `pick up the {物体名}`（如 `pick up the mustard bottle`）。
   训练时与 LIBERO 一样经过配置里的 `CoT_prompt` 模板（由 VLM 接口统一套用）。
5. **混合训练**：新的 `cotrain_human_robot` 数据入口，把 LIBERO 机器人 dataloader 和人手 dataloader 按固定节奏交错（`human_every: N` 表示每 N 个 batch 中有 1 个人手 batch），每个 batch 仍是单一域。
   - 人手 batch：只算几何（waypoint）+ 阶段 + 物体损失；动作头和 SpatialAdapter 不产生梯度。
   - 日志：分别缓存两个域最近一次的指标，人手指标加 `human/` 前缀，每次日志都输出两者，避免交错节奏和日志间隔对齐导致某个域的指标从不出现。
   - `eval_action_model` 只在机器人 batch 上执行。
   - 只支持单进程训练（与现有实验一致，`STARVLA_DISABLE_DEEPSPEED=1`）；多卡 DDP 下人手 batch 会有未使用参数，需另行处理。

## 方案 3：阶段监督 + 物体轨迹监督

两个辅助头都接在 WaypointModel 的输出特征上（每个 tau slot 一个），默认权重为 0 时不创建，旧 checkpoint 可严格加载。

1. **阶段监督（两个域共享）**：4 类 `approach / grasp / transport / release`，对每个未来 slot 预测其所处阶段。
   - 机器人（LIBERO，gripper 动作 1=张开、0=闭合）：闭合期间为 transport；闭合指令前 `phase_window_s`（默认 0.4 s）内为 grasp；张开指令后 `phase_window_s` 内为 release；其余为 approach。
   - 人手（DexYCB）：被抓物体相对第一帧位移超过 1 cm 记为开始搬运（transport）；之前 `phase_window_s` 内为 grasp；其余为 approach。DexYCB 序列在举起后结束，没有 release。
   - 机器人阶段由 dataset 在线从 episode 的 gripper 动作列计算，通过 `datasets.vla_data.phase_labels: true` 开启，默认关闭。
   - 损失：交叉熵，按有效 slot 掩码，权重 `spatial.phase_weight`。
2. **物体轨迹监督（仅人手）**：预测被抓物体中心在**当前手腕坐标系**下的位置（当前时刻 + 各未来 slot），与执行者无关。
   - 标签来自 DexYCB `pose_y[ycb_grasp_ind]`（相机系），转到桌面系后再转到当前手腕系；未来帧使用与 waypoint 相同的时间重标定。
   - LIBERO 没有物体位姿标签，机器人 batch 上该损失被掩码为 0。后续可从仿真回放中提取物体位姿补上。
   - 损失：Huber（位移 /0.1 m，与 waypoint 位移损失同尺度），权重 `spatial.object_weight`。
3. 当前不把阶段/物体预测输入 SpatialAdapter（会改变 `spatial_projection` 输入维度，破坏旧 checkpoint 兼容）。若辅助监督有效，再作为后续实验。

## 新增 / 修改文件

| 文件 | 内容 |
|---|---|
| `examples/human2robots/DexYCBSpatial/prepare_dexycb_cotrain.py` | 生成 v2 sidecar：时间重标定的 waypoint、语言、阶段、物体轨迹 |
| `starVLA/dataloader/human_wrist_dataset.py` | 读取 v2 sidecar 的可选字段（语言、阶段、物体） |
| `starVLA/dataloader/cotrain_dataloader.py` | 机器人/人手交错 dataloader |
| `starVLA/dataloader/__init__.py` | `cotrain_human_robot` 入口 |
| `starVLA/dataloader/geometry_lerobot_dataset.py` | 可选的机器人阶段标签 |
| `starVLA/model/framework/VLM4A/QwenSpatialOFT.py` | arm_id、anchor_mode、阶段/物体头与损失 |
| `starVLA/training/train_starvla.py` | 分域指标缓存、eval 跳过人手 batch |
| `examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_bridge_spatial_cotrain.yaml` | 方案 1+2 |
| `examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_bridge_spatial_cotrain_phase.yaml` | 方案 1+2+3 |
| `tests/test_spatial_cotrain.py` | 单元测试 |

## 实验设计

所有实验与 Bridge-Spatial only 保持相同：`libero_spatial_oft_long2x`、50K 步、batch 32、相同学习率；评估用 `eval_libero_checkpoint_sweep.sh`（8 worker、batch 4、seed 7、NVIDIA EGL），10K–50K 全部评估。

| 实验 | 配置 | 对照 |
|---|---|---|
| A. 混合训练 | 方案 1+2，`human_every: 4`，阶段/物体权重 0 | Bridge-Spatial only（92.00% @50K） |
| B. 混合训练 + 阶段/物体监督 | 方案 1+2+3，`phase_weight: 0.1`，`object_weight: 0.5` | A |

判定标准：标准 LIBERO 平均分标准误约 0.6 pp，Long 约 1.8 pp。只有平均分差 > 1.2 pp 或 Long 差 > 3.6 pp（约 2σ）才视为有效差异。

注意：混合训练每 4 个 batch 有 1 个人手 batch，50K 步中机器人 batch 只有 37.5K。若 A 的结果不如对照，需要再加一组“50K 机器人 batch + 额外人手 batch”（总步数 66.7K）排除步数因素。

## 实施状态（2026-10-27）

- [x] 代码实现（模型、数据、训练循环、两份配置）
- [x] 单元测试：`tests/test_spatial_cotrain.py` 12 项通过；`test_spatial_mem_v2/v3`、`test_config_overrides` 37 项无回归
- [x] v2 sidecar 生成：`playground/DerivedData/dexycb_cotrain_v2`（train 49,846 / val 7,149 / test 5,553 样本，20 种物体）
  - 时间重标定后 train 中位位移 0.0070/0.0138/0.0278/0.0557 m，LIBERO 0.0064/0.0128/0.0255/0.0502 m（差距从 1.5–1.9 倍降到约 1.1 倍）
  - 当前时刻阶段分布（train）：approach 30.8%、grasp 11.2%、transport 58.0%
- [x] GPU 冒烟测试：两份配置各跑 12 步（batch 4），机器人/人手指标、阶段/物体损失均正常输出，checkpoint 正常保存
- [ ] 实验 A（GPU 0）/ B（GPU 1）并行训练 50K 步 + 10K–50K 标准 LIBERO 评估：`run_cotrain_experiments.sh`，日志在 `logs/bridge_spatial_cotrain_<时间>/`
- 实验 C（OFT + long2x 对照）按 2026-10-27 决定不做

配置说明：两份配置的 `eval_interval` 为 0（Bridge-Spatial only 为 100），因为训练中的动作评估会从交错数据流里额外取一个 batch，打乱 1:3 的节奏。该评估不参与训练，不影响可比性。

## 待验证 / 风险

- `gravity` anchor 丢掉了高度信息；若 LIBERO 性能下降，可尝试加入“相对桌面的高度”（需要每个场景的桌面高度）。
- `time_scale` 只校准了位移中位数，未校准旋转速度。
- 人手图像只有一个第三人称视角，LIBERO 有主视角 + 腕部两个视角，两个域的图像 token 数不同。
- 人手阶段边界（物体开始移动）比机器人（夹爪指令）略晚，存在系统性偏差。
