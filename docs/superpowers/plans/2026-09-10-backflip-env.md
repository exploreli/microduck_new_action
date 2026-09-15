# 后空翻环境 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development
> （推荐）或 superpowers:executing-plans 按任务逐条实现本计划。步骤用 checkbox
> （`- [ ]`）语法跟踪。本计划对应的实现**已完成并通过验证**，故步骤标记为 `- [x]`
> 并附实测结果。

**目标：** 新增 RL 任务 `Mjlab-Backflip-Flat-MicroDuck`，让 microduck（groundcontact
模型）从站立下蹲、起跳、团身，在**自由飞行**中向后转满 2π，再展体、**用双脚落地站稳**。
一次性 episodic 策略，部署时热切换触发，无相位时钟、无参考轨迹。

**架构：** 一个新 env cfg，从 mjlab 的 `make_velocity_env_cfg()` 出发（与 standup /
roulade 相同基座），继承 robot / DR / obs-noise / 延迟 / NaN 守卫栈，删除步行奖励，
嫁接 backflip 奖励集。核心是**接触反转（contact-inverted）的空中门控旋转累加器**：
只在无地面接触时积分向后 pitch（`-ω_y`），因此地上向后滚一分不得——这是把「后空翻 =
腾空」的物理本质编码进进度计量的反作弊设计。所有新奖励函数进 `src/mjlab_microduck/tasks/mdp.py`。

**技术栈：** Python 3.12、mjlab 1.3.0、MuJoCo / mujoco-warp、torch 2.9.1、rsl_rl（PPO）、uv、pytest。

**参考 spec：** `docs/superpowers/specs/2026-09-10-backflip-env-design.md`

## 全局约束

- **符号约定**：`mdp.py` 混用两种——自取负惩罚（返回 ≤ 0，如 `backflip_stand_tax`）
  用**正**权重；正幅度成本（返回 ≥ 0，如 `backflip_overspeed_penalty`）用**负**权重。
  铁律：每次 run 每个 `Episode_Reward/<penalty>` 必须 ≤ 0。
- **obs actor 必须保持 61 维**（与 walking / standup / roulade 逐字节一致），否则 ONNX
  无法在 runtime 槽热切换。不用的命令槽（head 4 / body 6）**零填充**，绝不删槽。
- **关节永远按名字/正则经 `_servo_joint_ids` / `_servo_joint_pos` 解析，绝不硬编码索引**。
  groundcontact 模型无 passive 关节（servo 视图为 identity），但仍走 helper 以保持正确。
- **空中门传感器名 `robot_ground_contact` 是 LOAD-BEARING**：`mdp.py` 的
  `_BACKFLIP_AIR_SENSOR` 必须与 env cfg 注册的整机接触传感器名逐字一致。
- **没有无门控的 |a_z| 惩罚**：起跳是大 +a_z、落地是大 -a_z，`trunk_vertical_accel_penalty`
  会税翻转赖以成功的机制。settle 改由 height 门控的 `arrival_damping` 后期塑造。
- **motion-blockers（`body_ang_vel` / `angular_momentum`）保持 ≈0**：翻转就是大角速度事件。
- **支付速率上限 / 超速阈值必须高于自然翻转速率（~12–16 rad/s）**：18 / 22 rad/s。
- **`ENABLE_SYMMETRY = True`**：后空翻是矢状面对称动作，镜像损失对抗侧手翻塌陷。
- **BAM 执行器**：standalone env 必注册 `expand_bam_friction_fields`（startup）；
  joint-friction DR 走 `randomize_bam_friction`（scale `friction_scale`），因为 BAM 下
  `dof_frictionloss` 被清零，随机化它是静默 no-op。
- 代码注释风格：env cfg / mdp 的 backflip 段用**英语**（对齐 roulade / standup 兄弟代码）。
- 测试命令：`uv run --with pytest pytest tests/ -q`（本机因网络受限，用
  `.venv/bin/python` + 独立运行器执行，见 Task 4）。

---

### Task 1：`mdp.py` 的 backflip MDP 函数段

空中门控累加器 + 全部 backflip 奖励/事件函数。这是整个任务的核心逻辑。

**文件：**
- 修改：`src/mjlab_microduck/tasks/mdp.py`（新增 backflip 段，~7250–7668）

**接口：**
- 复用：`com_upward_velocity`、`body_ang_vel_at_height`、`standing_composite_score`、
  `_lateral_axis_z`、`_sensor_any_contact`、`_servo_joint_ids` / `_servo_joint_pos`
- 产出：
  - 常量 `_BACKFLIP_ROT_SIGN = -1.0`、`_BACKFLIP_AIR_SENSOR = "robot_ground_contact"`
  - `_backflip_state` / `_backflip_airborne` / `_update_backflip_accum` / `_backflip_completion_gate`
  - `reset_backflip_state(...)`（事件）
  - `backflip_progress`、`backflip_landing_composite`、`backflip_upright_after_flip`、
    `backflip_height_after_flip`、`backflip_landing_sharp`、`backflip_stand_tax`、
    `backflip_overspeed_penalty`、`backflip_flatness_penalty`、`backflip_sagittal_penalty`、
    `backflip_lateral_velocity_penalty`

- [x] **Step 1：实现累加器与空中门**

`_update_backflip_accum`：`airborne = ~robot_ground_contact`；`omega_bf = _BACKFLIP_ROT_SIGN * root_link_ang_vel_b[:,1]`；
`delta = nan_to_num(omega_bf)·step_dt·airborne`，再乘矢状面 flatness smoothstep 门；
更新 `accum` / `max`（前沿单调）/ `airborne_latch`；step-guarded（同一步多奖励项不重复积分）。

- [x] **Step 2：实现 `reset_backflip_state`（站立 + 空中两类 spawn）**

空中 spawn：`pitch = -θ`（θ ∈ [120°,330°]）、`qvel[:,4] = _BACKFLIP_ROT_SIGN·ω`、
`qvel[:,2] = vz`、trunk z ∈ [0.13,0.19]、HOME→TUCK lerp（`tuck_factor ∈ [0.4,1.0]`）、
累加器/max/paid 预置到 spawn 角、latch=True。**必须排在 `reset_robot_joints` 之后**。

- [x] **Step 3：实现进度与落地/惩罚奖励**

`backflip_progress`（potential-based 前沿增量 + `max_paid_rate` 封顶）；落地四件套
（composite 乘积 / upright / height / sharp）+ `backflip_stand_tax`（自取负）均门控在
`_backflip_completion_gate`（前沿 smoothstep + airborne latch）；overspeed / sagittal /
flatness / lateral_velocity 返回正幅度（配负权重）。

- [x] **Step 4：验证函数签名与既有 helper 一致**

对照 `reset_roulade_state` 确认 qpos/qvel 列约定：`qpos[2]=z`、`qpos[3:7]=quat`、
`qvel[4]=body ω_y`、关节列 `= 7 + servo_ids[idx]`。

---

### Task 2：环境配置文件

**文件：**
- 创建：`src/mjlab_microduck/tasks/microduck_backflip_env_cfg.py`（733 行）

**接口：**
- 消费：Task 1 的全部函数 + `MICRODUCK_STANDUP_ROBOT_CFG`、`SYMMETRY_CFG`、
  `PpoWithSymmetryCfg`、`VelocityCommandCommandOnlyCfg`、`zero_command_padding`
- 产出：`make_microduck_backflip_env_cfg(play=False, rough=False)`、`MicroduckBackflipRlCfg`

- [x] **Step 1：顶部常量 + ENABLE_* 开关**

`EPISODE_LENGTH_S=4.0`、`STAND_Z=0.115`、`BACKFLIP_AIR_PITCH_MIN/MAX=radians(120/330)`、
`AIR_Z_MIN/MAX=0.13/0.19`、`AIR_OMEGA_RANGE=(4.0,9.0)`、`AIR_VZ_RANGE=(-0.2,0.4)`、
`LANDING_GATE_LO/HI=radians(300/355)`、`MAX_PAID_RATE=18.0`、`OVERSPEED_OMEGA_MAX=22.0`、
`TUCK_OVERRIDES`、`_LEG_JOINTS`/`_NECK_JOINTS`、`ENABLE_SYMMETRY=True`。

- [x] **Step 2：三个接触传感器 + robot 装配**

`feet_ground_contact` / `self_collision` / `robot_ground_contact`（空中门）。
`cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}`，`action.scale=1.0`。

- [x] **Step 3：删步行奖励，嫁接 18 项 backflip 奖励集**

删 `track_linear_velocity` / `track_angular_velocity` / `air_time` / `foot_clearance` /
`foot_swing_height` / `foot_slip` / `pose` / `upright` / `soft_landing`；按 spec 的奖励表
设置权重（progress 8.0、landing_composite 6.0、stand_tax +5.0、overspeed -0.05 等）。

- [x] **Step 4：obs 61D（删 base_lin_vel/height_scan，零填充命令槽，IMU DR + _safe critic）**

- [x] **Step 5：命令中和 + 终止项（只留 nan_state + timeout）**

`twist` → `VelocityCommandCommandOnlyCfg`（微噪声）；删 `fell_over`（摔倒就是任务本身）；
`nan_state` 带 `sensor_names=(feet_ground_contact,)`。

- [x] **Step 6：事件（BAM 展开 + set_backflip_state + DR）与 6 项课程**

`expand_bam_friction_fields`（startup）、`reset_action_history`、`set_backflip_state`
（reset，排在 `reset_robot_joints` 后）、COM/head-COM/armature/mass-inertia/joint-friction
DR；删 `push_robot`。课程：`backflip_spawn_mix` / `com_range` / `head_com_range` /
`action_rate_weight` / `arrival_damping_weight` / `torque_rate_weight`。

- [x] **Step 7：`MicroduckBackflipRlCfg`**

hidden_dims (512,256,128)、obs_normalization True、`PpoWithSymmetryCfg` +
`symmetry_cfg=SYMMETRY_CFG`、`experiment_name="microduck_backflip"`、`num_steps_per_env=24`、
`max_iterations=10_000`。

---

### Task 3：注册任务

**文件：**
- 修改：`src/mjlab_microduck/tasks/__init__.py`

- [x] **Step 1：导入 + 注册**

```python
from .microduck_backflip_env_cfg import (
    make_microduck_backflip_env_cfg,
    MicroduckBackflipRlCfg,
)
# ...
register_mjlab_task(
    task_id="Mjlab-Backflip-Flat-MicroDuck",
    env_cfg=make_microduck_backflip_env_cfg(),
    play_env_cfg=make_microduck_backflip_env_cfg(play=True),
    rl_cfg=MicroduckBackflipRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

- [x] **Step 2：验证注册**

`uv run list-envs`（本机 `.venv/bin/list-envs`）输出含 `Mjlab-Backflip-Flat-MicroDuck`
（索引 1）。**已通过。**

---

### Task 4：配置回归测试

**文件：**
- 创建：`tests/test_backflip_cfg.py`（521 行，36 个测试）

- [x] **Step 1：编写 36 个 CPU 测试（仿 `tests/test_roller_standup_cfg.py`）**

分组：构建/形状、关节索引在真实 groundcontact 模型上解析、奖励卫生（无步行奖励、
无无门控 |a_z|、符号约定锁、自碰撞轻、motion-blockers≈0）、任务特定（progress
potential-based、支付速率/超速高于自然翻转、落地门控、STAND_Z=0.115、composite 目标）、
airborne 门（传感器注册名、累加器向后符号 -1）、spawn/课程（set_backflip_state 排在
base reset 后、空中 spawn 真腾空、spawn mix 偏向站立、无 fall 终止、nan_state 盯接触
传感器）、命令/obs（twist 中和、command-only、与 roulade 的 61D 奇偶校验、命令槽零填充）、
DR（BAM 展开注册、继承 DR 事件/课程存活、后期打磨课程起步 0、action_rate ramp 保持轻）、
对象标识/对称（trunk asset cfg 是独立对象、矢状面对称启用、RL cfg 烘焙 normalizer 且
experiment_name 独立）。

- [x] **Step 2：运行测试**

本机网络受限（pytest 不在 uv 缓存且无法下载），因测试全为无 pytest 特性的零参数纯函数，
用 `.venv/bin/python` + importlib 独立运行器执行。**结果：36 passed, 0 failed / 36 total（EXIT=0）。**

---

### Task 5：仿真验证（冒烟 run + ONNX 导出）

单元测试不证明 env 真能跑：61D obs、接触传感器、关节解析、无 NaN 只有启动仿真才看得到。
这是计划的出口门。

- [x] **Step 1：验证 actor obs 真实维度 = 61**

冒烟日志：`Active Observation Terms in Group: 'actor' (shape: (61,))`，critic `(74,)`，
action `(14)`。**通过。**

- [x] **Step 2：64 envs / 5 iters 冒烟训练（带 NaN 守卫）**

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
.venv/bin/train Mjlab-Backflip-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5
```

结果：5 次迭代无异常、无 NaN（`nan_state: 0.0000` 每迭代）；18 项 `Episode_Reward/`
全部计算且**每个 penalty ≤ 0**；`Mean symmetry loss: 0.0288`；课程 stage 0 正确
（spawn_mix 0.5、action_rate -0.1、arrival/torque 0.0）；保存 `model_0.pt` / `model_4.pt`。

Mean reward 从 -0.73 降到 -2.97 而 episode length 从 14.26 升到 45.61——这是**冒烟测试
的正常签名**：随机策略下 episode 变长、每步惩罚累积，5 iters 尚未发现翻转。判据是
「构建/无 NaN/obs 61D/每项计算/ONNX 导出」，不是奖励上升。**通过。**

- [x] **Step 3：ONNX 导出（normalizer 烘焙）**

```bash
.venv/bin/python scripts/export.py Mjlab-Backflip-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_backflip/<run>/model_4.pt \
  --onnx-file logs/smoke_backflip.onnx
```

结果：`onnx.checker` 通过，opset 18，**INPUT `obs [1,61]` → OUTPUT `actions [1,14]`**，
Actor MLP `61→512→256→128→14` 且 `EmpiricalNormalization` 已烘焙，部署元数据齐全
（action_scale / joint_names / observation_names / run_path 等）。**通过。**

- [x] **Step 4：把测量结果记进 spec**

见 `docs/superpowers/specs/2026-09-10-backflip-env-design.md` 的「验证结果（初始）」章节。

---

## 给实现者的备注

- **顺序**：Task 1 → 2 → 3 → 4 → 5。Task 2 消费 Task 1 的全部函数；Task 3 消费 Task 2；
  Task 4/5 验证前四者。
- **本任务的 1 号陷阱**是旋转累加器的**门控方向**：roulade 是 contact-GATED（支撑滚翻），
  backflip 必须是 contact-INVERTED（空中翻转）。搞反了，policy 会在地上向后滚刷满 2π
  而从不跳起来。`test_accumulator_sign_is_backward` 与 `test_mid_air_spawns_are_genuinely_airborne` 守这条。
- **2 号陷阱**是符号约定：`backflip_stand_tax` 自取负 → **正**权重（5.0）。给它负权重会
  双负号变成「奖励瘫成一堆」。`test_already_negative_penalties_use_positive_weights` 守这条。
- **3 号陷阱**是 61D obs 奇偶校验：任何偏离 roulade 布局的改动都会让 ONNX 在 runtime 槽
  不可用。`test_obs_parity_with_roulade_env` 守这条。
- **4 号陷阱**是 motion-blockers / |a_z|：翻转就是大角速度 + 大垂直加速度事件，任何有意义
  的权重或无门控的 |a_z| 都会在动作被发现前阻断它。`test_motion_blockers_near_zero_during_discovery`
  与 `test_no_ungated_vertical_accel_penalty` 守这条。
- **本计划范围外**：完整训练（数千 iters @ 4096 envs）、正式 play 观察、runtime 部署。
  命令见 spec；计划止于 env 被验证为「可构建、可步进、可导出」（Task 5）。
- **本机环境备注**：uv 缓存属主/权限曾阻塞依赖同步，解法是可写缓存
  `export UV_CACHE_DIR=/home/dev_common/.cache/uv` + `uv sync --offline`；网络受限（~36 KB/s）
  时训练/导出直接用 `.venv/bin/*`（已 sync，289 包），避免 uv 触发在线解析。
