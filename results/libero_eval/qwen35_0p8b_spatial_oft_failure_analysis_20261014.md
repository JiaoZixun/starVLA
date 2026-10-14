# Qwen3.5-0.8B Spatial OFT: Long Failure Analysis

## Result and comparison scope

The Spatial OFT checkpoint completed 2,000 LIBERO episodes and reached 388/500 (77.6%) on LIBERO-10. The 0.8B OFT baseline reached 381/500 (76.2%) with the same seed and episode count, a difference of +7 successes (+1.4 percentage points). Approximate 95% Wilson intervals are 73.7–81.0% and 72.3–79.7%, respectively; this is a small single-seed difference, not evidence by itself that spatial supervision caused the gain.

The repository reports 93.8% Long for StarVLA-OFT (Qwen3-VL). That published result is 81 successes out of 500 above the current checkpoint (+16.2 percentage points). The local 4B checkpoint/config is not a like-for-like control: it uses Qwen3-VL-4B, a DiT-B action head (hidden size 2560), `libero_all`, and a run configured for 80K maximum steps; the local checkpoint evaluated here is the 50K-step checkpoint. The README's published result is labeled 30K steps / 9.54 epochs. Current Spatial OFT uses Qwen3.5-0.8B, an MLP action head (hidden size 1024), a waypoint head and spatial adapter, `libero_spatial_oft_all`, and 50K steps. The two mixtures both contain all four LIBERO suites, so Long task exposure is not the obvious distinction. Backbone capacity, action-head architecture, optimization, checkpoint step, and evaluation compatibility are confounded. A controlled scale claim requires a matched 0.8B/4B training and evaluation setup.

The local 4B diagnostics should not be read as a reproduction of the paper's model result. One 5-episode Object smoke run scored 0/5. A separate one-episode action diagnostic also failed; its saved trajectory contains implausibly large action components (for example, a 2.32-unit component in one action chunk). The diagnostics used an isolated historical checkout and its own eval interface. They indicate unresolved local checkpoint/interface compatibility, not evidence against the published 4B result.

## Long task outcomes

| Task | Spatial OFT | 0.8B OFT baseline | Change | Failure-relevant structure |
|---|---:|---:|---:|---|
| 0: alphabet soup + tomato sauce into basket | 32/50 (64%) | 34/50 (68%) | -2 | Two sequential pick-and-place operations |
| 1: cream cheese + butter into basket | 48/50 (96%) | 46/50 (92%) | +2 | Two sequential pick-and-place operations |
| 2: turn on stove + place moka pot | 45/50 (90%) | 41/50 (82%) | +4 | State change followed by placement |
| 3: bowl into bottom drawer + close | 46/50 (92%) | 45/50 (90%) | +1 | Place, then articulated-object interaction |
| 4: mugs on left/right plates | 37/50 (74%) | 35/50 (70%) | +2 | Identity, relation, and sequential placement |
| 5: book into back caddy compartment | 46/50 (92%) | 39/50 (78%) | +7 | Small target region / depth relation |
| 6: mug on plate + pudding to its right | 32/50 (64%) | 28/50 (56%) | +4 | Second object and relational placement |
| 7: alphabet soup + cream cheese into basket | 41/50 (82%) | 41/50 (82%) | 0 | Two sequential pick-and-place operations |
| 8: both moka pots on stove | 24/50 (48%) | 31/50 (62%) | -7 | Repeated grasp and placement of similar objects |
| 9: yellow/white mug into microwave + close | 37/50 (74%) | 41/50 (82%) | -4 | Object placement followed by door closure |

Tasks 0, 6, 8, and 9 account for 75 of the 112 Spatial OFT Long failures. Long is not uniformly worse under Spatial OFT: task 5 improved by seven successes, while task 8 lost seven and task 9 lost four. The pattern points to remaining weakness in multi-stage execution, object/state tracking, and phase transitions, with the largest regression on repeated similar-object handling. It does not establish that every failure is a grasp-alignment failure.

## Replay inspection

The full Spatial OFT evaluation did not save videos. To inspect failure modes, the same checkpoint, suite, seed, client, and server were used to rerun tasks 0, 6, 8, and 9 until each produced one success and one failure. These are diagnostic examples, not a random sample and not a failure-rate estimate.

| Task | Observed in the failure replay | Interpretation limit |
|---|---|---|
| 0 | The target remains outside the basket through the captured trajectory; the robot stays around the target area instead of completing the two-object sequence. | The agent-view video does not reveal contact forces, gripper contact, or whether an object was briefly lifted and dropped. |
| 6 | The first mug placement appears completed; the required second-stage pudding placement is not completed in the failure example. | The camera does not establish whether this is a failed grasp, bad relative placement, or a phase-switch/state-tracking error. |
| 8 | The trajectory involves attempts around both moka pots and the stove; the failure example does not finish the required two-pot placement. | A single camera view cannot distinguish missed contact, object slip, collision, or a mistaken next-object selection. |
| 9 | The mug/microwave sequence proceeds into the articulated door interaction, but the episode ends without satisfying the whole task. | The replay alone cannot determine whether the mug pose or the door-closing action caused the terminal failure. |

These examples are consistent with the user's observation that an unsuccessful grasp may be followed by later actions without a deliberate retry. The current client requests an 8-action chunk and re-observes only after consuming it. It has no explicit object-held check or recovery state machine. This makes within-chunk correction impossible and leaves recovery to the learned policy at the next observation. However, the current replays do not prove that this mechanism caused every shown failure. A quantitative claim needs simulator object pose, end-effector pose, gripper state/contact, and per-step action/chunk-boundary logs for all sampled failures.

## Paper figure proposal

Use a paired, simulator-grounded **spatial grounding and recovery figure**, not a raw VLM attention overlay as the main evidence.

1. **Matched trials:** compare baseline OFT and Spatial OFT at identical task, initial-state index, seed, camera views, and action execution settings. Include both successes and failures, sampled across multiple seeds. At minimum use Long tasks 0, 6, 8, and 9; add task 5 as a positive example.
2. **Four event-aligned frames:** approach/grasp, gripper closure, post-grasp lift, and placement/phase transition. Show the front and wrist images, simulator-rendered target/object/receptacle masks, gripper state, and event labels. Draw the end-effector trajectory and the Spatial OFT predicted SE(3) waypoints after transforming them from the current end-effector frame into world coordinates and projecting them into both cameras.
3. **Causal visual evidence:** overlay patch-level Integrated Gradients for the predicted waypoint translation (and separately the gripper/action output) on the RGB input. Keep the input, target attribution, and color scale fixed between models. Attention weights alone are not a causal explanation and can be misleading; use them only as a supplementary view if desired.
4. **Quantitative panels:** report target-mask attribution mass, grasp-point/waypoint translation error in centimeters, orientation error in degrees, object lift success, first-attempt grasp success, recovery/retry rate after detected failure, placement success, and final task success. Bootstrap confidence intervals should resample episodes, clustered by task and seed.
5. **Ablations:** compare matched 0.8B OFT vs Spatial OFT, then Spatial OFT with the waypoint-to-action adapter removed or waypoint inputs shuffled. This distinguishes a spatial-supervision effect from backbone scale and tests whether the spatial branch changes control in a useful way.

The primary figure should support a narrow, defensible claim such as: “Spatial OFT changes where visual evidence is used for waypoint prediction and improves geometric grounding/recovery on selected matched tasks.” Only make the broader claim that it improves spatial perception if the attribution, waypoint-error, and outcome metrics all support it. A heatmap by itself cannot establish that spatial information improved perception.

## Artifacts

- Full Spatial OFT evaluation: `results/libero_eval/qwen35_0p8b_spatial_oft_full_20261014_1348/evaluation_report.md`
- Baseline report and baseline replay notes: `docs/starvla_qwen35_0p8b_oft_libero_50k_baseline.md`
- Spatial OFT matched-task diagnostic replays: `results/libero_eval/qwen35_0p8b_spatial_oft_failure_review_20261014/libero_10/`
- Public benchmark table: `examples/simBenchmarks/LIBERO/README.md`
