# StarVLA Qwen3.5-0.8B OFT 50K LIBERO 基线评估

> 记录日期：2026-10-12。模型为 50K step checkpoint；全量评估按 seed 7 运行一次。

## 结论

四套 LIBERO benchmark 共完成 **1802/2000（90.1%）**。Spatial、Object、Goal 分别为 92.2%、97.8%、94.2%；LIBERO-10 为 76.2%，是当前主要短板。此前 10 任务各 5 次的小批验证为 90.5%，与全量结果接近。

## Checkpoint 与评估配置

- Checkpoint：`playground/Checkpoints/qwen35_0p8b_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt`（1,748,288,891 bytes）。
- 模型 / OFT：Qwen3.5-0.8B，OFT action head；action chunk 为 8，归一化统计 key 为 `franka`。训练配置为 `examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_oft_libero.yaml`。
- 评估卡：NVIDIA A100-PCIE-40GB，GPU 1；NVIDIA EGL / A100 renderer 检查通过，未使用 OSMesa。
- 四个 suite 各 10 个任务、每任务 50 次，seed 7，共 2000 个 episode；视频保存关闭。
- 并行设置：8 个仿真 worker、推理服务最大 batch 4、收集窗口 12 ms；full-run 周期日志显示平均 batch 约 1.51。服务端固定 CPU 32–35，worker 固定 CPU 36–43。
- 当前评估 client 使用训练匹配的 PIL bicubic resize，服务端按 `franka` 统计反归一化动作。
- 全量运行时间约 71 分钟（14:43–15:54 CST）。

## 全量结果

| Suite | 成功 | 成功率 | 任务数 | 结果文件 |
|---|---:|---:|---:|---|
| Spatial | 461/500 | 92.2% | 10 | [summary.json](../results/libero_eval/qwen35_0p8b_oft_50k_parallel_full_20261012/libero_spatial/summary.json) |
| Object | 489/500 | 97.8% | 10 | [summary.json](../results/libero_eval/qwen35_0p8b_oft_50k_parallel_full_20261012/libero_object/summary.json) |
| Goal | 471/500 | 94.2% | 10 | [summary.json](../results/libero_eval/qwen35_0p8b_oft_50k_parallel_full_20261012/libero_goal/summary.json) |
| LIBERO-10 | 381/500 | 76.2% | 10 | [summary.json](../results/libero_eval/qwen35_0p8b_oft_50k_parallel_full_20261012/libero_10/summary.json) |
| **总体** | **1802/2000** | **90.1%** | **40** | — |

### 逐任务结果

#### Spatial

| Task ID | 任务 | 成功 | 成功率 |
|---:|---|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 50/50 | 100.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 42/50 | 84.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 49/50 | 98.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 47/50 | 94.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 49/50 | 98.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 47/50 | 94.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 44/50 | 88.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 42/50 | 84.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 44/50 | 88.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 47/50 | 94.0% |

#### Object

| Task ID | 任务 | 成功 | 成功率 |
|---:|---|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 49/50 | 98.0% |
| 1 | pick up the cream cheese and place it in the basket | 50/50 | 100.0% |
| 2 | pick up the salad dressing and place it in the basket | 50/50 | 100.0% |
| 3 | pick up the bbq sauce and place it in the basket | 49/50 | 98.0% |
| 4 | pick up the ketchup and place it in the basket | 50/50 | 100.0% |
| 5 | pick up the tomato sauce and place it in the basket | 49/50 | 98.0% |
| 6 | pick up the butter and place it in the basket | 50/50 | 100.0% |
| 7 | pick up the milk and place it in the basket | 48/50 | 96.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 44/50 | 88.0% |
| 9 | pick up the orange juice and place it in the basket | 50/50 | 100.0% |

#### Goal

| Task ID | 任务 | 成功 | 成功率 |
|---:|---|---:|---:|
| 0 | open the middle drawer of the cabinet | 48/50 | 96.0% |
| 1 | put the bowl on the stove | 50/50 | 100.0% |
| 2 | put the wine bottle on top of the cabinet | 50/50 | 100.0% |
| 3 | open the top drawer and put the bowl inside | 42/50 | 84.0% |
| 4 | put the bowl on top of the cabinet | 44/50 | 88.0% |
| 5 | push the plate to the front of the stove | 49/50 | 98.0% |
| 6 | put the cream cheese in the bowl | 46/50 | 92.0% |
| 7 | turn on the stove | 50/50 | 100.0% |
| 8 | put the bowl on the plate | 50/50 | 100.0% |
| 9 | put the wine bottle on the rack | 42/50 | 84.0% |

#### LIBERO-10

| Task ID | 任务 | 成功 | 成功率 |
|---:|---|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 34/50 | 68.0% |
| 1 | put both the cream cheese box and the butter in the basket | 46/50 | 92.0% |
| 2 | turn on the stove and put the moka pot on it | 41/50 | 82.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 45/50 | 90.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 35/50 | 70.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 39/50 | 78.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 28/50 | 56.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 41/50 | 82.0% |
| 8 | put both moka pots on the stove | 31/50 | 62.0% |
| 9 | put the yellow and white mug in the microwave and close it | 41/50 | 82.0% |

## 推理请求延迟

原 full-eval 日志没有单次推理延迟，因此在 GPU 1 补跑了计时样本：每个 suite 选 Task ID 0–4，每个任务 10 个 episode，共 200 个 episode。固定 seed 7，仍使用 8 个 worker、最大 batch 4、12 ms batching window。

`latency` 的边界是 LIBERO client 调用 `WebsocketClientPolicy.predict_action` 到收到并解码响应，代表一个 8-step action chunk 的请求往返时间。计时包含 client 校验/序列化、WebSocket、server batching 等待、模型推理和响应处理；不包含观测预处理及 MuJoCo 仿真 step。每个 suite 均按该 suite 全部请求数加权计算平均值。

| Suite | 采样成功 | 推理请求数 | 平均请求延迟 |
|---|---:|---:|---:|
| Spatial | 45/50 (90.0%) | 721 | 176.53 ms |
| Object | 50/50 (100.0%) | 873 | 180.96 ms |
| Goal | 46/50 (92.0%) | 804 | 184.31 ms |
| LIBERO-10 | 37/50 (74.0%) | 1986 | 143.42 ms |
| **总体** | **178/200 (89.0%)** | **4384** | **163.84 ms** |

计时 server 的周期日志平均 batch 为 1.45（最大 batch 4）；请求延迟反映了 8-worker 并行负载，不是单请求、无排队的纯模型 kernel 时间。

计时采样结果和逐 shard 的 `inference_timing.json`、`summary.json` 位于：

`results/libero_eval/qwen35_0p8b_oft_50k_inference_latency_20261012/`

## 小批量验证结果

正式全量前按 10 个任务、每任务 5 次完成了 200 个 episode 验证。

| Suite | 成功 | 成功率 |
|---|---:|---:|
| Spatial | 48/50 | 96.0% |
| Object | 50/50 | 100.0% |
| Goal | 46/50 | 92.0% |
| LIBERO-10 | 37/50 | 74.0% |
| **总体** | **181/200** | **90.5%** |

## 成功与失败回放视频

全量基线运行时关闭了视频保存。之后在 GPU 1 使用同一 checkpoint、seed 7 对每个 suite 选取一个任务补采视频，每个任务最多 10 次；同一任务一旦分别拿到成功和失败视频便提前停止。

| Suite | Task ID | 补采结果 | 成功视频 | 失败视频 |
|---|---:|---:|---|---|
| Spatial | 1 | 10/10 成功；达到 10 次上限后停止 | [成功视频](../results/libero_eval/qwen35_0p8b_oft_50k_success_failure_videos_20261012/libero_spatial/task_1/rollout_pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_success.mp4) | 未采到 |
| Object | 8 | 10/10 成功；达到 10 次上限后停止 | [成功视频](../results/libero_eval/qwen35_0p8b_oft_50k_success_failure_videos_20261012/libero_object/task_8/rollout_pick_up_the_chocolate_pudding_and_place_it_in_the_basket_success.mp4) | 未采到 |
| Goal | 3 | 5 次（4 次成功、1 次失败），拿到两类视频后停止 | [成功视频](../results/libero_eval/qwen35_0p8b_oft_50k_success_failure_videos_20261012/libero_goal/task_3/rollout_open_the_top_drawer_and_put_the_bowl_inside_success.mp4) | [失败视频](../results/libero_eval/qwen35_0p8b_oft_50k_success_failure_videos_20261012/libero_goal/task_3/rollout_open_the_top_drawer_and_put_the_bowl_inside_failure.mp4) |
| LIBERO-10 | 0 | 2 次（1 次成功、1 次失败），拿到两类视频后停止 | [成功视频](../results/libero_eval/qwen35_0p8b_oft_50k_success_failure_videos_20261012/libero_10/task_0/rollout_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_success.mp4) | [失败视频](../results/libero_eval/qwen35_0p8b_oft_50k_success_failure_videos_20261012/libero_10/task_0/rollout_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_failure.mp4) |

Spatial 和 Object 在各自 10 次上限内都只出现成功，因此按上限停止，没有追加任务或重跑。视频补采统计仅描述被选任务，不改变上面的全量基线分数。

## 结果解释与注意事项

- 全量 90.1% 与小批验证 90.5% 一致，未见新代码评估链路造成整体分数异常下降的迹象。
- LIBERO-10 全量 76.2%，低于其余 suite；任务级最低项包括双物体放置、抽屉操作和多物体摆放。该项值得单独作为后续改进目标。
- 以上是单 seed（7）基线，不是多 seed 均值。推理延迟是并行负载下的 action-chunk 请求往返延迟，不应解释为 episode 时长或纯 GPU kernel latency。
- 个别 worker 在结果写出并正常退出后会打印 robosuite EGL context 析构阶段的 `EGL_NOT_INITIALIZED` 清理告警；所有 worker 都完成并生成了 summary，且 NVIDIA EGL 预检通过。
