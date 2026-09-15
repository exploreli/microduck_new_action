"""Microduck BACKFLIP task — backward aerial flip, land back on the feet.

Episodic policy: robot starts standing, crouches, jumps, tucks and rotates
BACKWARD through a full 2π in free flight, then untucks and lands on its feet.
Triggered at deployment like sit/standup/roulade (policy switch = flip starts
immediately; no phase clock, no reference motion).

Sibling of the roulade (see microduck_roulade_env_cfg.py and the roulade
section of mdp.py). The ONE physical difference reshapes the whole design:

  • A ROULADE is a SUPPORTED roll — it never leaves the floor, so its rotation
    accumulator is contact-GATED (ballistic flips earn nothing).
  • A BACKFLIP is an AERIAL flip — the rotation happens in free flight, so the
    accumulator is contact-INVERTED: it only integrates while NO robot geom
    touches the terrain. A backward roll/rock on the ground therefore earns no
    progress and never opens the landing gate; the ONLY way to accumulate flip
    progress is to actually jump and spin in the air.

Everything else reuses the proven roulade/standup episodic recipe (the full
rationale lives in the backflip section of mdp.py):
  • ONE dense progress signal — paid increments of the max-so-far cumulative
    backward rotation (potential-based; a full flip pays 2π total, camping pays
    zero per step), gated on being airborne.
  • Launch bootstrap — com_upward_velocity pays the take-off push while still
    low (capped, so it can't be farmed by bunny-hopping).
  • Landing rewards (composite product, upright, height, sharp, stand-tax)
    gated on FLIP COMPLETION (frontier ≥ ~300°) AND on an AIRBORNE LATCH —
    state-based gates, not a clock. "Do nothing" earns nothing; the standing
    spawn cannot farm them; only jumping AND flipping opens the annuity.
  • Reverse curriculum via MID-AIR spawns: a slice of episodes starts already
    airborne, tucked, pitched 120°–330° into the flip with backward angular
    momentum, accumulator pre-set to the spawn angle. The second half (untuck →
    spot the floor → land) is where a backflip is won or lost, so it gets dense
    on-policy data from iteration 0.

SPEED / IMPACT (AGENTS.md physics-alignment):
  • A 25 cm robot completes 2π in roughly its airtime (~0.4 s) → ~12–16 rad/s
    of pitch is NATURAL. The paid-rate cap (18 rad/s) and the overspeed penalty
    (22 rad/s) sit well above that so they tax only absurd whips, never the
    physically-necessary rotation (the roulade run-3 "cap forfeits the maneuver"
    lesson).
  • There is NO ungated |a_z| penalty: the take-off IS a large +a_z explosion
    and the landing a large -a_z, so trunk_vertical_accel_penalty would tax the
    very mechanism the flip needs. Post-landing settle is shaped by the
    height-gated arrival damper (body_ang_vel_at_height), introduced late.

DR / obs / regularisers mirror the standup/roulade env (velocity sim2real
parity), with the motion-blockers (body_ang_vel, angular_momentum) kept near
zero during discovery — the flip IS a large angular-velocity event.
"""

import math
from copy import deepcopy

# Symmetry — the backflip is sagittal / left-right symmetric; the mirror loss
# directly fights the sideways-collapse (cartwheel) failure mode. Same 61-dim
# symmetry table the roulade uses.
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to standup/roulade for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a push mid-flip is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the standup/roulade env) ───────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: crouch + jump + flip (~1 s) + land + settle. 4 s leaves ~3 s of
# standing annuity after a completed flip without wasting compute.
EPISODE_LENGTH_S = 4.0

# Empirically-measured standing trunk height on the groundcontact model (the
# standup/roulade value for THIS exact model — don't carry it across models).
STAND_Z = 0.115

# ── Mid-air spawn (reverse curriculum) ────────────────────────────────────────
# θ = backward rotation already done at spawn. 90° = face-up horizontal, 180° =
# fully inverted (head straight down), 270° = face-down horizontal coming
# around, >300° opens the landing gate at birth → dense data on the last mile
# (untuck → land). Spawns are placed in FREE FLIGHT (z above standing) so the
# airborne accumulator keeps integrating from the first step.
BACKFLIP_AIR_PITCH_MIN   = math.radians(120.0)
BACKFLIP_AIR_PITCH_MAX   = math.radians(330.0)
BACKFLIP_AIR_Z_MIN       = 0.13   # trunk z above terrain — genuinely airborne
BACKFLIP_AIR_Z_MAX       = 0.19
BACKFLIP_AIR_OMEGA_RANGE = (4.0, 9.0)    # rad/s backward pitch at spawn
BACKFLIP_AIR_VZ_RANGE    = (-0.2, 0.4)   # m/s world-z (hang time), may be negative

# Tuck anchor: legs folded into a ball (knees toward chest) to spin fast. Same
# leg-fold as the roulade tuck; NO chin-tuck — a backflip throws the head BACK,
# not forward, so neck/head stay at HOME. Servo-index keyed; mid-air spawns lerp
# HOME→tuck by a per-env factor.
TUCK_OVERRIDES = {
    2:  -1.15,  # left  hip_pitch
    3:   1.25,  # left  knee
    4:   1.05,  # left  ankle
    11:  1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

# Rotation thresholds (rad) for the completion gate. The flip lands near 2π;
# the gate opens over the final ~55° so it is essentially "flip nearly done".
LANDING_GATE_LO = math.radians(300.0)
LANDING_GATE_HI = math.radians(355.0)

# Natural flip rate is ~12–16 rad/s; cap the PAID rate and the overspeed tax
# well above it so only absurd whips are forfeited/penalised.
MAX_PAID_RATE      = 18.0
OVERSPEED_OMEGA_MAX = 22.0

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck backward aerial flip environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot ground contact — the AIRBORNE GATE: the rotation accumulator
    # only integrates while NO robot geom touches the terrain, so a backward
    # ground roll earns no progress and never completes.
    # NAME IS LOAD-BEARING: _backflip_airborne reads it (mdp._BACKFLIP_AIR_SENSOR).
    robot_ground_cfg = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: backflip task set ────────────────────────────────────────────
    # Progress increments — the one dense task signal during the flip. Airborne-
    # gated + potential-based (a full flip pays 2π total, camping pays 0/step).
    cfg.rewards["backflip_progress"] = RewardTermCfg(
        func=microduck_mdp.backflip_progress,
        weight=8.0,
        params={"target_angle": 2 * math.pi, "max_paid_rate": MAX_PAID_RATE},
    )

    # Launch bootstrap — pay the take-off push (upward CoM velocity) while still
    # low, capped so it can't be farmed by bunny-hopping and only fires below
    # the standing target (once airborne/high it pays nothing). Not recovery-
    # gated (gate_z_below=None): we WANT it during the clean crouch-jump.
    cfg.rewards["backflip_launch"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=1.0,
        params={
            "asset_cfg":  SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": STAND_Z + 0.03,
            "max_vz":     1.5,
        },
    )

    # Completion-gated standing annuity — the dominant attractor. Broad stds
    # (standup composite lesson: a partial landing must score visibly, ~0.2+).
    cfg.rewards["backflip_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_composite,
        weight=6.0,
        params={
            "target_height":    STAND_Z,
            "height_std":       0.04,
            "upright_std":      0.40,
            "pose_std":         0.40,
            "joint_indices":    _LEG_JOINTS,
            "gate_lo":          LANDING_GATE_LO,
            "gate_hi":          LANDING_GATE_HI,
            "target_overrides": None,
            "asset_cfg":        SceneEntityCfg("robot"),
        },
    )

    # Completion-gated bootstrap layers (gradient far from the goal, where the
    # composite product is ≈0): linear upright + broad height Gaussian.
    cfg.rewards["backflip_upright_after_flip"] = RewardTermCfg(
        func=microduck_mdp.backflip_upright_after_flip,
        weight=2.0,
        params={
            "gate_lo":   LANDING_GATE_LO,
            "gate_hi":   LANDING_GATE_HI,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["backflip_height_after_flip"] = RewardTermCfg(
        func=microduck_mdp.backflip_height_after_flip,
        weight=1.5,
        params={
            "target_height": STAND_Z,
            "std":           0.04,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
            "asset_cfg":     SceneEntityCfg("robot"),
        },
    )

    # Sharp landing layer: tight-std upright × height product on top of the
    # broad composite so the policy finishes tall instead of parking at the
    # touchdown crouch (the two-layer last-mile lesson).
    cfg.rewards["backflip_landing_sharp"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_sharp,
        weight=3.0,
        params={
            "target_height": STAND_Z,
            "height_std":    0.015,
            "upright_std":   0.3,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
            "asset_cfg":     SceneEntityCfg("robot"),
        },
    )

    # Completion-gated stand tax (THE standup lesson): once the flip is done,
    # every step spent below STAND_Z costs — "land then crumple in a heap"
    # flips from free to net-negative. Gate closed during the flip, so the flip
    # itself is never taxed; mid-air spawns are born with it active.
    # NOTE: SELF-NEGATING (returns -shortfall·gate) → POSITIVE weight.
    cfg.rewards["backflip_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.backflip_stand_tax,
        weight=5.0,
        params={
            "target_height": STAND_Z,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
            "asset_cfg":     SceneEntityCfg("robot"),
        },
    )

    # ── Flip-plane shaping ────────────────────────────────────────────────────
    # Whip-speed tax (generous threshold — a real flip is ~12–16 rad/s).
    cfg.rewards["backflip_overspeed"] = RewardTermCfg(
        func=microduck_mdp.backflip_overspeed_penalty,
        weight=-0.05,
        params={"omega_max": OVERSPEED_OMEGA_MAX},
    )
    # Keep the rotation in the sagittal plane (a cartwheel is not a backflip):
    # out-of-plane ω, lateral drift, and lateral-axis tilt.
    cfg.rewards["backflip_sagittal"] = RewardTermCfg(
        func=microduck_mdp.backflip_sagittal_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["backflip_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.backflip_lateral_velocity_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["backflip_flatness"] = RewardTermCfg(
        func=microduck_mdp.backflip_flatness_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Motion-blockers stay near zero during discovery (the flip IS a large
    # angular-velocity event); the settle/polish pressure comes from the
    # LATE-introduced gated terms below (arrival_damping, torque rate).
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # must stay ≈0: the flip is ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # Arrival damper — trunk ω_xy² gated on standing height AND low tilt, so the
    # flip itself is never taxed; introduced at 0 and ramped by curriculum.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    0.09,
            "height_high":   0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Self-collision — LIGHT: a tucked flip needs body-on-body contact (knees
    # against trunk); standup's -1.0 would fight the tuck.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Always-on upright would oppose the flip (the core failure of the old
    # attempts); landing uprightness is handled by the completion-gated terms.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical 61D layout to walking / standup / roulade) ────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # No terrain ray sensor in this env → drop the height_scan / foot_height obs.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Hard landings can spike contact forces to non-finite a step before qpos
    # diverges; the critic's sensor-derived terms are the one obs path nan_state
    # can't protect, and a single NaN there kills the run (2026-08-21 crash).
    # Sanitize them (critic-only, costs the policy nothing).
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots: zero padding for BOTH head (4) and body (6) — the flip
    # uses no commanded pose, but the 61D obs layout parity with the rest of the
    # family is kept so the runtime stack hot-swaps this policy unchanged.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # Falling over is the task — keep only the NaN guard + timeout. sensor_names
    # lets the guard also catch a non-finite contact force from a hard landing.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Standing start + mid-air reverse-curriculum spawns; also resets the
    # rotation accumulator (must run after reset_robot_joints — dict insertion
    # order — since the mid-air tuck lerps FROM the HOME pose it wrote).
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_state,
        mode="reset",
        params={
            "standing_prob":     0.5,
            "air_prob":          0.5,
            "standing_z_min":    0.11,
            "standing_z_max":    0.12,
            "standing_tilt_max": math.radians(5.0),
            "air_pitch_min":     BACKFLIP_AIR_PITCH_MIN,
            "air_pitch_max":     BACKFLIP_AIR_PITCH_MAX,
            "air_z_min":         BACKFLIP_AIR_Z_MIN,
            "air_z_max":         BACKFLIP_AIR_Z_MAX,
            "air_omega_range":   BACKFLIP_AIR_OMEGA_RANGE,
            "air_vz_range":      BACKFLIP_AIR_VZ_RANGE,
            "tuck_overrides":    TUCK_OVERRIDES,
            "tuck_factor_range": (0.4, 1.0),
            "joint_noise_std":   0.08,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Reverse-curriculum mix: heavy mid-air early (the landing sub-task is
    # learnable from day 0 — it overlaps the standup face-up recovery), shift
    # toward standing starts as the full jump+flip gets discovered. Mid-air
    # never goes to zero: it keeps the last mile practiced and is realistic DR.
    cfg.curriculum["backflip_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_backflip_state",
            "param_stages": [
                {"step": 0,          "params": {"standing_prob": 0.50, "air_prob": 0.50}},
                {"step": 3000 * 24,  "params": {"standing_prob": 0.65, "air_prob": 0.35}},
                {"step": 6000 * 24,  "params": {"standing_prob": 0.80, "air_prob": 0.20}},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # action_rate ramp — smoothness polish, kept light so it never blocks the
    # fast flip; introduced from step 0 at -0.1 and tightened late.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 2000 * 24,  "weight": -0.2},
                {"step": 4000 * 24,  "weight": -0.4},
            ],
        },
    )

    # Settle polish — introduced only after the flip skill exists (the standup
    # timing lesson: any attempt-tax active during discovery prevents the
    # maneuver from being found at all; the fix is timing, not magnitude).
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -0.025},
                {"step": 3500 * 24,  "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -5e-4},
                {"step": 3500 * 24,  "weight": -1e-3},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckBackflipRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_backflip",
    run_name="microduck_backflip",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
