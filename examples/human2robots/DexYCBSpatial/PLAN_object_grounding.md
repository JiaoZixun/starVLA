# 物体 grounding（Exp C）实施记录与结果

分支：`libero-object-grounding`（基于 `libero-hand-cotrain-phase` 的 Exp A 配置）
日期：2026-10-28（实现、训练）/ 2026-10-29（评估、失败分析）

## 方案

在 Exp A（人手/机器人混合训练，`human_every: 4`，`anchor_mode: gravity`）基础上：

- **P0 物体 grounding token**：WaypointModel 新增 2 个 object slot（0 = 下一个要操作的物体，1 = 放置位置），与 waypoint slot 一起解码，
  输出当前末端坐标系下的 xyz（0.1 m 单位）和主视角图像 uv。SpatialAdapter 读取**预测的** grounding 作为额外 key token。
  两个域都监督（Huber，`grounding_xyz_weight: 0.5`，`grounding_uv_weight: 0.5`）。
  - 机器人标签（`libero_grounding_v1`）：LeRobot demo 没有物体位姿，用夹爪交互点代替。目标 = 下一次闭合时的 TCP（闭合期间为 TCP 本身），
    放置点 = 该次松开时的 TCP；最后一次松开之后标签无效。
  - 人手标签（`dexycb_grounding_v3`）：物体中心（`pose_y`）；每个样本配一个反事实样本（同一图像，"pick up the <另一个物体>"，只监督该物体位置）。
- **P1 训练时图像增强**：主视角相似变换（uv 标签同步变换）+ 光度扰动，副视角 15% 概率丢弃（`grounding_augment.py`）。
- **P2 二值夹爪标签**：`gripper_label: closure`，两个域都用"是否闭合"（机器人用指令，人手用物体开始移动）。
- 训练 30K 步（warmup 3K），batch 32，单卡；其余与 Exp A 相同。

## 新增 / 修改文件

| 文件 | 内容 |
|---|---|
| `examples/simBenchmarks/LIBERO/train_files/prepare_grounding_sidecars.py` | LIBERO grounding sidecar + 相机投影审计（NVIDIA EGL） |
| `examples/human2robots/DexYCBSpatial/prepare_dexycb_grounding.py` | DexYCB v3 sidecar（双视角、物体中心、反事实） |
| `starVLA/dataloader/grounding_augment.py` | 标签一致的图像增强 |
| `starVLA/dataloader/geometry_lerobot_dataset.py`、`human_wrist_dataset.py`、`__init__.py` | 读取 grounding 标签、closure 夹爪标签、增强 |
| `starVLA/model/framework/VLM4A/QwenSpatialOFT.py` | object slot、grounding 头与损失、adapter 中的 object token |
| `examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_bridge_spatial_grounding.yaml` | Exp C 配置 |
| `examples/human2robots/DexYCBSpatial/run_grounding_experiment.sh` | 训练 + 10K/20K/30K 评估（failure tracing 开启） |
| `tests/test_object_grounding.py` | 单元测试 |

## 结果（标准 LIBERO，8 worker、batch 4、seed 7、NVIDIA EGL）

| 运行 | Step | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|---:|
| Exp C grounding | 10K | 30.6 | 48.6 | 52.0 | 16.4 | 36.90 |
| Exp C grounding | 20K | 90.8 | 92.8 | 87.4 | 48.4 | 79.85 |
| Exp C grounding | 30K | 97.6 | 93.8 | 92.8 | 73.4 | **89.40** |
| Spatial-only（2026-10-29 重跑，tracing 开） | 30K | 95.8 | 99.2 | 90.2 | 83.2 | 92.10 |
| Exp A cotrain（2026-10-29 重跑，tracing 开） | 50K | 95.6 | 98.2 | 94.2 | 82.6 | 92.65 |

两组基线重跑与原结果相差 ≤ 3 个 episode，可复现。Exp C 比两组基线低约 2.7–3.3 pp，Long 低约 9–10 pp，超出 2σ 判定阈值。
注意：Exp C 30K 是学习率衰减完的终点，Spatial-only 30K 处于 50K 计划中途，两者训练进度不完全对等。

结果目录：`results/qwen35_0p8b_bridge_spatial_grounding_libero_30k/checkpoint_sweep_20261029_014410/`，
基线：`results/libero_eval/*_libero_std_20261029_131402/`。

## 失败分析（`LIBERO_FAILURE_TRACE=1`）

主要差异是**新出现的"抓着物体停住、不松手"**：

| 4 个 suite 合计 | Exp C | Spatial-only | Exp A |
|---|---:|---:|---:|
| 失败 episode | 212 | 158 | 147 |
| 停住（stalled） | 112 | 45 | 38 |
| 停住时仍抓着物体 | 51 | 7 | 6 |
| 其中曾达成目标但没松手 | 17 | 0 | 0 |

按任务：

- **Long 任务 4（白杯放左盘、黄白杯放右盘）**：1/50（Spatial-only 40/50）。42 个 episode 抓着白杯直到超时；黄白杯只在 2/50 个 episode 中被抓过（基线 44/50）。
- **Long 任务 3（碗放进抽屉并关上）**：31/50（基线 46/50）。19 个失败中 18 个是碗已放好后停住，抽屉没关。
- **Long 任务 6（白杯放盘上、布丁放盘右）**：27/50（基线 42/50）。布丁基本都抓到，但放置时停住，夹爪反复开合（中位 12 次）。
- **Long 任务 8（两个 moka 壶）**：三个模型都约 37/50，与 Exp C 无关。
- **Object 任务 0（alphabet soup）**：26/50（基线 50/50）。成功 episode 的下降抓取点 x≈-0.136（std 0.7 cm）；
  失败 episode 下降点偏 8–13 cm，落在另一个罐子上方，没有抓取。

**grounding 头本身是准的**：在 libero_10 训练 demo 上离线测试 30K checkpoint（任务 4/6/8，各 6 个 episode，步长 4），
第一个物体松开后 slot 0 的预测会切换到下一个物体（中位误差 2.6–4.7 cm，而此时目标距夹爪 16–30 cm），闭合期间误差约 1 cm。
因此问题出在闭环执行，而不是 grounding 标签或预测。

## 结论与下一步

- Exp C 不如两组基线，不作为后续基线。
- 原因尚未定位。Exp C 相对 Exp A 同时改了增强、`gripper_label: closure`、object slot、DexYCB v3 + 反事实、30K 计划。
  失败集中在"该松手时不松"，首要怀疑：
  1. `gripper_label: closure`：夹爪监督从连续 qpos 改成二值闭合指令。
  2. object token：闭合期间 slot 0 标签恒为零（目标就是夹爪自身），最后一次松开后没有监督，但推理时 adapter 仍读取该 token。
- 建议的最小消融：只把 `gripper_label` 改回 `qpos`，其余不变，重训 30K（单卡约 8 h，评估约 4.5 h）。
