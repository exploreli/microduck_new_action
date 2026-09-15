# 设计 — `backflip`：后空翻（腾空中翻转，落回双脚）

日期：2026-09-10。Task id：`Mjlab-Backflip-Flat-MicroDuck`。

> 本文档用中文撰写（应用户要求）；对应的代码注释（`microduck_backflip_env_cfg.py`、
> `mdp.py` 的 backflip 段）沿用仓库既有惯例保持**英语**，与 `roulade` / `standup`
> 兄弟环境一致。

## 目标

一个新的 RL 任务：让 microduck（步行/groundcontact 模型，非 rollers）完成一个
**后空翻**——从站立开始，下蹲、起跳、抱膝团身，在**自由飞行**中向后转满 2π，
然后展体、看清地面、**用双脚落地**并站稳。

与 `sit` / `standup` / `roulade` 一样是 **episodic（一次性）策略**：部署时通过
策略热切换触发，切换即开始翻转——**没有相位时钟，没有参考动作轨迹**。整条轨迹
（下蹲多深、怎么甩臂蹬腿、何时展体）都交给 PPO 自己发现。

## 决策定案

| 问题 | 决策 |
|---|---|
| 支撑模型 | `MICRODUCK_STANDUP_ROBOT_CFG`（`robot_groundcontact.xml`，14 舵机，无 passive 关节） |
| 触发方式 | 策略热切换，一次性；无相位、无参考轨迹 |
| 旋转方向 | **向后**（body-frame 负 ω_y），`_BACKFLIP_ROT_SIGN = -1.0` |
| 进度计量 | 空中门控（contact-inverted）旋转累加器，只积自由飞行中的 -ω_y |
| 主奖励 | 单一稠密信号：max-so-far 空中后翻角前沿的**增量**（potential-based） |
| 落地奖励 | 状态门控（前沿 ≥ ~300°）**且**要求 airborne latch——不是时钟 |
| 逆向课程 | 一部分 episode 从**空中**（已团身、已翻 120°–330°）开始 |
| 对称性 | `ENABLE_SYMMETRY = True`——后空翻是矢状面/左右对称动作 |
| 时长 | 4.0 s（下蹲+跳+翻 ~1 s + 落地 + ~3 s 站立年金） |

## 与 roulade 的核心物理差异（这一条重塑了整个设计）

后空翻是 `roulade`（前滚翻）的兄弟环境，但**唯一的物理差异**决定了全套奖励结构：

- **ROULADE 是支撑滚翻**——它从不离开地面，所以它的旋转累加器是
  **接触门控（contact-GATED）**：只有轮子/头还在地上转才算数，弹道式腾空翻转
  一分不得。
- **BACKFLIP 是空中翻转**——旋转发生在自由飞行中，所以累加器是
  **接触反转（contact-INVERTED）**：只有当**没有任何 robot geom 接触地面**时才
  积分。因此在地上向后滚/向后摇都累积不到进度、也永远打不开落地门；**唯一**
  能积累翻转进度的方式，就是真的跳起来、在空中转。

这条差异就是本环境最核心的**反作弊**设计：它把「向后翻」这个动作的物理本质
（腾空）直接编码进了进度计量的门控里，而不是靠一个小惩罚去劝。

## 采用的方法与理由

沿用 roulade / standup 已验证的 episodic 配方（完整理由见 `mdp.py` 的 backflip 段）：

- **单一稠密进度信号**——支付 max-so-far 累计后翻角的增量（potential-based：
  翻满一圈总共支付 2π，原地 camp 每步支付 0），且空中门控。
- **起跳 bootstrap**——`com_upward_velocity` 在还低的时候支付起跳的向上推，
  带上限（`max_vz=1.5`、只在 `max_height=STAND_Z+0.03` 以下生效），所以没法
  靠 bunny-hop（原地蹦）刷分。
- **落地奖励**（composite 乘积、upright、height、sharp、stand-tax）门控在
  **翻转完成**（前沿 ≥ ~300°）**且** airborne latch 上——都是基于状态的硬门，
  不是时钟。「什么都不做」一无所获；站立 spawn 也无法刷这些奖励；只有**跳起来
  并翻转**才能解锁这份年金。
- **逆向课程（空中 spawn）**——一部分 episode 一开始就在空中：已团身、已后翻
  120°–330°、带向后角动量、累加器预置到 spawn 角。后空翻的成败关键在于**后半程**
  （展体 → 看清地面 → 落地），所以从第 0 次迭代起就给这段稠密的 on-policy 数据。

## 架构

**文件**：`src/mjlab_microduck/tasks/microduck_backflip_env_cfg.py`

- factory `make_microduck_backflip_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`
- PPO 配置 `MicroduckBackflipRlCfg`
- task id `Mjlab-Backflip-Flat-MicroDuck`，在 `tasks/__init__.py` 注册

**基座**：从 mjlab 的 `make_velocity_env_cfg()` 出发（与 standup / roulade 相同），
因此免费继承 base_com / dof_pos_limits / out_of_terrain_bounds、基础 reset 事件、
obs 噪声/延迟栈。步行专属奖励被删除，backflip 奖励集被嫁接进来。

**三个接触传感器**：

| 传感器 | 用途 |
|---|---|
| `feet_ground_contact` | 双脚触地（nan_state 守卫也读它） |
| `self_collision` | 自碰撞成本（团身需要身体接触，故权重很轻） |
| `robot_ground_contact` | **空中门控**——`_BACKFLIP_AIR_SENSOR`，NAME IS LOAD-BEARING，必须与 `mdp.py` 常量逐字一致 |

## 核心机制：空中门控旋转累加器

`mdp.py` 的 `_update_backflip_accum(env, asset)`（每个控制步只积分一次，step-guarded）：

```python
airborne  = _backflip_airborne(env)                                  # = NOT robot_ground_contact
omega_bf  = _BACKFLIP_ROT_SIGN * asset.data.root_link_ang_vel_b[:, 1]  # -ω_y（向后为正）
delta     = nan_to_num(omega_bf) * step_dt * airborne.float()          # 只有腾空才积分
# 矢状面 flatness 门（与 roulade 共享）：侧翻/肩翻把横轴甩离水平 → 折扣/清零
y_z = |lateral_axis_z(root_link_quat_w)|
t   = clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0, 1)      # 0.866 / 0.5
delta = delta * smoothstep(t)
accum = accum + delta
max   = maximum(max, accum)                                            # 前沿只前进，不倒退
airborne_latch |= airborne                                             # 一旦离地就永久解锁
```

要点：

- **`_BACKFLIP_ROT_SIGN = -1.0`**：向后翻 = body-frame **负** ω_y，取负后变成正的进度。
- **前沿（max）单调**：起跳前的下蹲后摇（wind-up rock）既不支付、也不会「撤销」已付。
- **flatness 门**：一个侧向/肩膀着地的「侧手翻」不是后空翻，横轴偏离水平会被 smoothstep
  折扣到 0，只有干净的矢状面翻转才计数。
- **airborne latch**：落地年金额外要求「本 episode 曾经离地」，所以一个从未真正腾空的
  小跳收不到站立奖励。

`backflip_progress` 支付的是前沿增量，并对**支付速率**封顶：

```python
new_paid = clamp(max_accum, max=2π)
delta    = clamp(new_paid - clamp(paid, max=2π), min=0)      # 只支付前进
delta    = clamp(delta, max=max_paid_rate * step_dt)         # 18 rad/s 上限
reward   = delta / (step_dt * 2π)                            # 翻满一圈总支付 = 1.0/步的积分
```

无可刷之处：直立 camp（0/步）、在前沿以下摇（0/步）、地上向后滚（空中门 → 0）、
翻过 2π（clamp）。上限 18 rad/s **高于**自然翻转速率（~12–16 rad/s），所以只会
没收荒谬的甩鞭，绝不会没收物理上必需的旋转（这是 roulade run-3「上限没收了整个
动作」的教训）。

## Rewards

### 符号约定（曾坑掉四个环境的那类 bug）

`mdp.py` 混用两种约定：

- **自取负**的惩罚（返回 ≤ 0，如 `backflip_stand_tax` 返回 `-shortfall·gate`）→ 用
  **正**权重。
- **正幅度**的成本（返回 ≥ 0，如 `backflip_overspeed_penalty`）→ 用**负**权重。

铁律：每次 run 里，wandb 上每个 `Episode_Reward/<penalty>` 都必须 ≤ 0。（本环境已在
冒烟测试中逐项核对，见「验证结果」。）

### backflip 任务奖励集（18 项，含继承的 dof_pos_limits）

| Reward | 权重 | 函数 | 说明 |
|---|---|---|---|
| `backflip_progress` | **8.0** | `backflip_progress` | 主信号：空中后翻前沿增量，target=2π，max_paid_rate=18 |
| `backflip_launch` | 1.0 | `com_upward_velocity` | 起跳 bootstrap：trunk 向上速度，max_height=0.145、max_vz=1.5 |
| `backflip_landing_composite` | **6.0** | `backflip_landing_composite` | 落地主吸引子：height×upright×pose 高斯**乘积**，宽 std（0.04/0.40/0.40），门控 |
| `backflip_landing_sharp` | 3.0 | `backflip_landing_sharp` | 锐利层：紧 std（height 0.015、upright 0.3）乘积，逼它「站直」而非停在落地蹲 |
| `backflip_upright_after_flip` | 2.0 | `backflip_upright_after_flip` | 完成门控的线性 upright bootstrap（远离目标处乘积≈0 时提供梯度） |
| `backflip_height_after_flip` | 1.5 | `backflip_height_after_flip` | 完成门控的宽 height 高斯 bootstrap（std=0.04） |
| `backflip_stand_tax` | **5.0（正）** | `backflip_stand_tax` | 自取负：翻完后每步低于 STAND_Z 都扣分——「落地后瘫成一堆」由免费变净负 |
| `backflip_overspeed` | -0.05 | `backflip_overspeed_penalty` | 甩鞭税：|ω| > 22 rad/s 才罚（远高于自然翻转） |
| `backflip_sagittal` | -0.1 | `backflip_sagittal_penalty` | 保持旋转在矢状面（侧手翻不是后空翻）：出平面 ω |
| `backflip_lateral_vel` | -0.5 | `backflip_lateral_velocity_penalty` | 横向漂移 |
| `backflip_flatness` | -0.5 | `backflip_flatness_penalty` | 横轴偏离水平的倾角 |
| `action_rate_l2` | -0.1 | `mdp.action_rate_l2` | 平滑度，课程升到 -0.4（始终轻，绝不阻断快速翻转） |
| `joint_torque_rate_l2` | 0.0 | `joint_torque_rate_l2` | 起步 0，课程后期升到 -1e-3（打磨） |
| `body_ang_vel` | -0.002 | `mdp.body_angular_velocity_penalty` | trunk_base；**必须≈0**：翻转本身就是大角速度事件 |
| `angular_momentum` | -0.001 | `mdp.angular_momentum_penalty` | **必须≈0**，同理 |
| `arrival_damping` | 0.0 | `body_ang_vel_at_height` | 落地后 settle 打磨：height 0.09–0.11 且低倾角门控的 trunk ω_xy²；起步 0，课程升到 -0.05 |
| `self_collisions` | -0.1 | `mdp.self_collision_cost` | **轻**：团身需要膝-躯干接触，standup 的 -1.0 会打架 |
| `dof_pos_limits` | -1.0（继承） | `mdp.joint_pos_limits` | 关节限位 |

**关键取舍**：

- **没有常开的 `upright`**——常开直立会与翻转直接对抗（这正是旧尝试的核心失败）；
  落地直立由完成门控的项处理。`upright` 从继承的奖励里被删掉。
- **没有无门控的 |a_z| 惩罚**——起跳本身就是一次大的 **+a_z** 爆发、落地是大的
  **-a_z**，`trunk_vertical_accel_penalty` 会去税翻转赖以成功的机制（「发现期不要税
  尝试」这一族教训）。落地后的 settle 改由 height 门控的 `arrival_damping` 塑造，
  且后期才引入。
- **motion-blockers（body_ang_vel / angular_momentum）保持≈0**——翻转就是大角速度
  事件，任何有意义的权重都会在动作被发现前就阻断它。settle/polish 的压力全部来自
  **后期引入的门控项**（arrival_damping、torque rate）。

## Reset / spawn（逆向课程）

`reset_backflip_state`（`mode="reset"`，事件名 `set_backflip_state`），**必须排在
`reset_robot_joints` 之后**（dict 插入顺序）——因为空中团身是从它写入的 HOME 姿态
lerp 过来的。

两类 spawn：

- **站立**（`standing_prob`）：z ∈ [0.11, 0.12]，tilt ≤ 5°，累加器清零，latch=False。
- **空中**（`air_prob`）：真正的自由飞行——
  - pitch θ ∈ [120°, 330°]（`qpos[3:7]` 四元数，绕 y 轴 `-θ`）
  - trunk z ∈ [0.13, 0.19]（**高于**站立，保证空中门从第一步就积分）
  - 向后角速度 ω ∈ [4.0, 9.0] rad/s（`qvel[4] = _BACKFLIP_ROT_SIGN * ω`）
  - 世界 z 速度 vz ∈ [-0.2, 0.4] m/s（滞空时间，可为负）
  - 团身：HOME → `TUCK_OVERRIDES` 按每 env 的 `tuck_factor ∈ [0.4, 1.0]` lerp
  - 累加器 / max / paid 预置到 spawn 角，latch=True

`TUCK_OVERRIDES`（servo-index keyed，只折**腿**，不低头——后空翻是把头向后甩，
颈/头保持 HOME）：

```python
{2: -1.15, 3: 1.25, 4: 1.05,     # 左 hip_pitch / knee / ankle
 11: 1.15, 12: -1.25, 13: -1.05}  # 右 hip_pitch / knee / ankle
```

θ 的语义：90° = 面朝上水平，180° = 完全倒立（头正下），270° = 面朝下水平转过来，
>300° 一出生就打开落地门 → 给「最后一公里」（展体 → 落地）稠密数据。

## Curriculum（6 项；步数 = iteration × 24）

| 课程 | 类型 | 阶段 |
|---|---|---|
| `backflip_spawn_mix` | `event_param_curriculum`（`set_backflip_state`） | 0: 站立 0.50/空中 0.50 → 3000·24: 0.65/0.35 → 6000·24: 0.80/0.20 |
| `com_range` | `com_range_curriculum`（`randomize_com`） | 0: 0.003 → 500·24: 0.005 → 1000·24: 0.01 → 1500·24: 0.015 |
| `head_com_range` | `com_range_curriculum`（`randomize_head_com`） | 0: 0.003 → 500·24: 0.005 → 1000·24: 0.01 |
| `action_rate_weight` | `reward_weight`（`action_rate_l2`） | 0: -0.1 → 2000·24: -0.2 → 4000·24: -0.4 |
| `arrival_damping_weight` | `reward_weight`（`arrival_damping`） | 0: 0.0 → 2500·24: -0.025 → 3500·24: -0.05 |
| `torque_rate_weight` | `reward_weight`（`joint_torque_rate_l2`） | 0: 0.0 → 2500·24: -5e-4 → 3500·24: -1e-3 |

删除的继承课程：`terrain_levels`、`command_vel`。

设计意图：早期重空中 spawn（落地子任务从第 0 天就可学，它与 standup 的面朝上恢复
重叠），随完整跳跃+翻转被发现，逐步偏向站立起跳。空中 spawn 永不为 0——它保持
「最后一公里」被练习，也是真实的 DR。settle/polish 税**只在翻转技能已存在之后**
才引入（standup 的时机教训：发现期任何「尝试税」都会让动作根本找不到；解药是**时机**，
不是幅度）。

## Observations（61D，与全家桶逐字节一致）

actor obs = 8 项 = **61D**（与 walking / standup / roulade 布局完全相同，保证 runtime
热切换）：

```
base_ang_vel(3) + projected_gravity(3) + joint_pos(14) + joint_vel(14)
+ actions(14) + command(3) + head_command(4) + body_command(6) = 61
```

- 删除 `base_lin_vel`（actor）、`height_scan`（无地形 ray 传感器）；critic 保留
  `base_lin_vel`（74D，特权 obs）。
- `head_command`(4) / `body_command`(6) 用 `zero_command_padding` **零填充**——后空翻
  不用命令姿态，但保留 obs 槽以维持 61D 契约。
- `twist` 命令被中和成零附近微噪声（`VelocityCommandCommandOnlyCfg`，lin ±0.01、
  ang_z ±0.05），只为 obs 形状存在。
- IMU DR：`base_ang_vel` / `projected_gravity` 走 misaligned 变体（6°），joint_vel
  延迟 1 步，encoder bias 打开（actor `biased=True`，critic `biased=False`）。
- critic 的 sensor 派生项（`foot_contact_forces`、`foot_air_time`）走 `_safe` 变体：
  硬落地可能在 qpos 发散前一步把接触力冲成非有限值，而这是 `nan_state` 唯一护不住
  的 obs 路径（2026-08-21 崩溃教训）。

## Domain randomization

与 standup / roulade 对齐（sim2real parity）：COM(0.003→0.015)、head COM(0.003→0.01)、
mass/inertia(0.95–1.05)、armature(0.9–1.1)、joint friction(0.9–1.1，走 BAM 的
`randomize_bam_friction` / `friction_scale`)、foot friction(0.7–1.3)、IMU 朝向(6°)、
encoder bias(±0.015)。

- **`ENABLE_VELOCITY_PUSHES = False`**：翻转中途推一把是不相干的，删除 `push_robot`。
- **kp/kd DR = OFF**（对齐 velocity）。
- **`expand_bam_friction_fields`（startup）必注册**：BAM 执行器下 `dof_frictionloss`
  被清零，joint-friction DR 必须 scale 执行器的 `friction_scale`，否则是静默 no-op。

## 速度 / 冲击的物理对齐（AGENTS.md）

一个 25 cm 的机器人在大约它的滞空时间（~0.4 s）内转满 2π → **~12–16 rad/s 的 pitch
是自然的**。因此：

- 支付速率上限 = **18 rad/s**（`MAX_PAID_RATE`）
- 超速税阈值 = **22 rad/s**（`OVERSPEED_OMEGA_MAX`）

两者都**远高于**自然翻转速率，只没收/惩罚荒谬的甩鞭，绝不碰物理上必需的旋转。
不要用人尺度的速度直觉去给一台 25 cm 机器人设转速上限。

## MDP 函数清单（`mdp.py`，backflip 段 ~7250–7668）

- 常量：`_BACKFLIP_ROT_SIGN = -1.0`、`_BACKFLIP_AIR_SENSOR = "robot_ground_contact"`
- 内部：`_backflip_state`（惰性建累加器状态）、`_backflip_airborne`、
  `_update_backflip_accum`、`_backflip_completion_gate`
- 事件：`reset_backflip_state`
- 奖励：`backflip_progress`、`backflip_landing_composite`、`backflip_upright_after_flip`、
  `backflip_height_after_flip`、`backflip_landing_sharp`、`backflip_stand_tax`、
  `backflip_overspeed_penalty`、`backflip_flatness_penalty`、`backflip_sagittal_penalty`、
  `backflip_lateral_velocity_penalty`
- 复用既有 helper：`com_upward_velocity`、`body_ang_vel_at_height`、
  `standing_composite_score`、`zero_command_padding`、`_lateral_axis_z`、`_sensor_any_contact`

关节解析走 `_servo_joint_ids` / `_servo_joint_pos`（groundcontact 模型无 passive 关节，
故为 identity，但绝不硬编码索引）；`_LEG_JOINTS = [0,1,2,3,4,9,10,11,12,13]`、
`_NECK_JOINTS = [5,6,7,8]` 是 servo 视图索引。

## RL runner 配置

`MicroduckBackflipRlCfg`：

- actor/critic MLP hidden_dims `(512, 256, 128)`，ELU，`obs_normalization=True`
  （normalizer 必须由 `scripts/export.py` 烘焙进 ONNX）
- `PpoWithSymmetryCfg` + `symmetry_cfg=SYMMETRY_CFG`（矢状面对称，镜像损失直接对抗
  侧向塌陷/侧手翻这一失败模式）；`symmetry loss` 用 61D 对称表
- clip 0.2、entropy 0.01、epochs 5、minibatches 4、lr 1e-3 adaptive、gamma 0.99、
  lam 0.95、desired_kl 0.01、max_grad_norm 1.0
- `experiment_name = "microduck_backflip"`、`num_steps_per_env = 24`、
  `save_interval = 250`、`max_iterations = 10_000`

预算：episodic 技巧约 1000 iters @ 4096 envs 起步；后空翻带逆向课程，预计需要
数千 iters 才能稳定翻+落。

## 验证结果（初始）

以下均为**实测**（2026-09-10），非估计。

### 环境构建

`make_microduck_backflip_env_cfg()` 成功构建：

- `Actuator config matched 14 joint(s)`（14 舵机正确匹配）
- actor obs group shape **(61,)**，critic **(74,)**，action **(14)**
- 18 奖励项、13 事件、6 课程、3 接触传感器

### CPU 配置回归测试

`tests/test_backflip_cfg.py`：**36 / 36 通过**（纯 CPU，无 sim）。锁定：符号约定、
airborne 门传感器注册名、累加器向后符号（-1）、空中 spawn 真腾空、与 roulade 的
61D obs 奇偶校验、关节索引在真实 groundcontact 模型上解析、无无门控 |a_z| 惩罚、
支付速率上限/超速阈值高于自然翻转、任务已注册等。

### 冒烟训练（64 envs / 5 iters，RTX 4090）

命令：`WANDB_MODE=disabled .venv/bin/train Mjlab-Backflip-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5`

- **无 NaN**：`Episode_Termination/nan_state: 0.0000`（每次迭代）；NaN-safe
  reward/advantage 补丁激活
- **18 奖励项全部计算**，符号约定全对——**每个 penalty 的加权值 ≤ 0**：
  `backflip_stand_tax -0.0000→-0.0734`、`backflip_sagittal -0.0430→-0.3634`、
  `backflip_flatness`、`backflip_lateral_vel`、`action_rate_l2`、`self_collisions`、
  `body_ang_vel`、`angular_momentum` 全 ≤ 0；attractor（`backflip_progress 0.0557`、
  `backflip_launch`、落地四项）全 ≥ 0
- **对称损失计算**：`Mean symmetry loss: 0.0288`
- **obs normalizer 就位**：actor/critic 均含 `EmpiricalNormalization()`
- **课程 stage 0 正确**：`backflip_spawn_mix 0.5`、`action_rate_weight -0.1`、
  `arrival_damping_weight 0.0`、`torque_rate_weight 0.0`
- checkpoint 保存：`model_0.pt`、`model_4.pt`

迭代轨迹（**属冒烟测试的正常签名，非 bug**）：

| iter | Mean reward | Mean episode length |
|---|---|---|
| 0 | -0.73 | 14.26 |
| 1 | -0.95 | 19.74 |
| 2 | -1.52 | 26.76 |
| 3 | -2.06 | 32.89 |
| 4 | -2.97 | 45.61 |

Mean reward 下降而 episode length 上升：随机策略下 episode 变长，每步惩罚
（sagittal / flatness / action_rate / stand_tax）随之累积；5 次迭代的随机策略尚未
发现翻转，正项接近 0。冒烟测试的判据是「构建 / 无 NaN / obs 61D / 每项计算 /
ONNX 导出」，**不是**奖励上升（那需要 1000+ iters）。

### ONNX 导出

命令：`.venv/bin/python scripts/export.py Mjlab-Backflip-Flat-MicroDuck --checkpoint-file <run>/model_4.pt --onnx-file logs/smoke_backflip.onnx`

- 导出成功，`onnx.checker.check_model` 通过，ir_version 8，opset 18，794 KB
- **INPUT `obs [1, 61]` → OUTPUT `actions [1, 14]`**（61D 契约 + 14 舵机）
- Actor MLP `61→512→256→128→14`，`obs_normalizer (EmpiricalNormalization)` **已烘焙**
- Critic MLP `74→512→256→128→1`
- 部署元数据齐全：`action_scale`、`command_names`、`default_joint_pos`、`joint_damping`、
  `joint_names`、`joint_stiffness`、`observation_names`、`run_path`

### 结论

配置层与导出层全部通过；后空翻是否**真的学得会**，需要一次 4096-env 的正式 run
（观察 `Episode_Reward/backflip_progress` 是否随迭代上升、空中 spawn 是否收敛到
翻满 2π 并落回双脚）。这是本设计文档交付时的下一步。
