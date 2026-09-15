"""Config-invariant tests for the Microduck BACKFLIP task (CPU only, no sim).

Mirrors tests/test_roller_standup_cfg.py conventions. Locks the AGENTS.md
invariants and the backflip-specific anti-cheat design:
  • reward sign convention (self-negating -> positive weight, magnitude -> negative)
  • the airborne gate sensor is registered under the load-bearing name
  • the rotation accumulator integrates BACKWARD pitch (_BACKFLIP_ROT_SIGN = -1)
  • the reverse-curriculum mid-air spawns are genuinely airborne
  • 61D obs parity with the roulade sibling (runtime hot-swap contract)
  • joint indices resolve on the ACTUAL groundcontact model
  • no ungated |a_z| penalty (the takeoff IS a +a_z explosion)
  • the paid-rate cap / overspeed tax sit ABOVE the natural flip rate
"""

import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    BACKFLIP_AIR_Z_MIN,
    EPISODE_LENGTH_S,
    LANDING_GATE_HI,
    LANDING_GATE_LO,
    MAX_PAID_RATE,
    OVERSPEED_OMEGA_MAX,
    STAND_Z,
    TUCK_OVERRIDES,
    MicroduckBackflipRlCfg,
    _LEG_JOINTS,
    _NECK_JOINTS,
    make_microduck_backflip_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)

# Walking/gait rewards that must NOT survive into an episodic aerial-flip env.
WALKING_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "pose",
    "upright",
    "soft_landing",
)


# ── Build & basic shape ───────────────────────────────────────────────────────


def test_env_builds_train_and_play():
    assert make_microduck_backflip_env_cfg() is not None
    assert make_microduck_backflip_env_cfg(play=True) is not None


def test_episode_is_short():
    # Episodic trick: crouch + jump + flip (~1 s) + land + a few s of standing
    # annuity. 4 s, like the other episodic maneuvers — not the 20 s walk default.
    cfg = make_microduck_backflip_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 4.0


def test_terrain_is_plain_plane():
    # A backflip is trained on flat ground; no terrain generator, no rough variant.
    cfg = make_microduck_backflip_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401  (import triggers registration)

    assert "Mjlab-Backflip-Flat-MicroDuck" in list_tasks()


# ── Joint indices resolve on the ACTUAL groundcontact model ───────────────────


def test_joint_indices_match_groundcontact_model():
    """Lock: _LEG_JOINTS / _NECK_JOINTS index the SERVO view of the real model.

    standing_composite_score consumes them via _servo_joint_pos (passive_*
    excluded). The groundcontact model has no passive joints, so the servo view
    is the identity on the articulated array — but the test still filters
    passive_* to stay correct if the model ever grows a passive jaw linkage.
    Pure CPU, no sim.
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_standup_spec

    model = get_standup_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    servo = [n for n in articulated if not n.startswith("passive_")]

    assert [servo[i] for i in _LEG_JOINTS] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    assert [servo[i] for i in _NECK_JOINTS] == [
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    ]


def test_tuck_overrides_target_leg_joints_only():
    # The tuck folds the LEGS (knees to chest) to spin fast; NO chin-tuck — a
    # backflip throws the head BACK, so neck/head stay at HOME. Servo-index keyed.
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_standup_spec

    model = get_standup_spec().compile()
    servo = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        and not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("passive_")
    ]
    leg_fold = {
        "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_pitch", "right_knee", "right_ankle",
    }
    assert set(TUCK_OVERRIDES) == {2, 3, 4, 11, 12, 13}
    for idx in TUCK_OVERRIDES:
        assert servo[idx] in leg_fold, f"tuck override {idx} -> {servo[idx]} is not a leg fold"


# ── Reward hygiene ────────────────────────────────────────────────────────────


def test_no_walking_rewards_survive():
    cfg = make_microduck_backflip_env_cfg()
    for name in WALKING_REWARDS:
        assert name not in cfg.rewards, f"walking reward survived: {name}"


def test_no_ungated_vertical_accel_penalty():
    """The takeoff IS a large +a_z explosion and the landing a large -a_z, so an
    ungated trunk_vertical_accel_penalty would tax the very mechanism the flip
    needs (the "don't tax attempts during discovery" family). Post-landing settle
    is shaped by the height-gated arrival damper instead.
    """
    cfg = make_microduck_backflip_env_cfg()
    assert "gentle_rise" not in cfg.rewards
    for name, term in cfg.rewards.items():
        assert term.func is not microduck_mdp.trunk_vertical_accel_penalty, (
            f"reward {name} uses an ungated |a_z| penalty that would tax the jump"
        )


def test_already_negative_penalties_use_positive_weights():
    """Sign-convention lock (the class of bug that bit four envs).

    mdp.py mixes two conventions: self-negating penalties (return <= 0) take a
    POSITIVE weight; positive-magnitude costs (return >= 0) take a NEGATIVE
    weight. Every Episode_Reward/<penalty> must read <= 0 in wandb.
    """
    cfg = make_microduck_backflip_env_cfg()
    # backflip_stand_tax returns -shortfall·gate (self-negating) -> POSITIVE weight.
    assert cfg.rewards["backflip_stand_tax"].weight > 0
    # Positive-magnitude penalties -> NEGATIVE weight.
    for name in (
        "backflip_overspeed",
        "backflip_sagittal",
        "backflip_lateral_vel",
        "backflip_flatness",
        "action_rate_l2",
        "self_collisions",
        "body_ang_vel",
        "angular_momentum",
    ):
        assert cfg.rewards[name].weight < 0, f"{name} must use a negative weight"
    # Attractor rewards (return >= 0) -> POSITIVE weight.
    for name in (
        "backflip_progress",
        "backflip_launch",
        "backflip_landing_composite",
        "backflip_upright_after_flip",
        "backflip_height_after_flip",
        "backflip_landing_sharp",
    ):
        assert cfg.rewards[name].weight > 0, f"{name} must use a positive weight"


def test_self_collision_is_light_for_tuck():
    # A tucked flip NEEDS body-on-body contact (knees against trunk); standup's
    # -1.0 would fight the tuck. Kept light.
    cfg = make_microduck_backflip_env_cfg()
    assert cfg.rewards["self_collisions"].weight == -0.1


def test_motion_blockers_near_zero_during_discovery():
    # The flip IS a large angular-velocity event: body_ang_vel / angular_momentum
    # must stay near zero or they block the maneuver before it is found.
    cfg = make_microduck_backflip_env_cfg()
    assert -0.01 <= cfg.rewards["body_ang_vel"].weight < 0.0
    assert -0.01 <= cfg.rewards["angular_momentum"].weight < 0.0
    # The settle/polish taxes start at ZERO and are ramped in late by curriculum.
    assert cfg.rewards["joint_torque_rate_l2"].weight == 0.0
    assert cfg.rewards["arrival_damping"].weight == 0.0


# ── Backflip task specifics ───────────────────────────────────────────────────


def test_backflip_progress_is_potential_based():
    # ONE dense signal: paid increments of the max-so-far airborne backward
    # rotation, up to 2π. Potential-based -> camping pays 0/step.
    cfg = make_microduck_backflip_env_cfg()
    params = cfg.rewards["backflip_progress"].params
    assert params["target_angle"] == 2 * math.pi
    assert params["max_paid_rate"] == MAX_PAID_RATE


def test_paid_rate_and_overspeed_above_natural_flip():
    """Physics alignment (AGENTS.md): a 25 cm robot completes 2π in ~0.4 s of
    airtime -> ~12-16 rad/s is NATURAL. The paid-rate cap and the overspeed tax
    must sit ABOVE that so they forfeit only absurd whips, never the flip itself
    (the roulade run-3 "cap forfeits the maneuver" lesson).
    """
    cfg = make_microduck_backflip_env_cfg()
    assert MAX_PAID_RATE >= 16.0
    assert OVERSPEED_OMEGA_MAX > MAX_PAID_RATE
    assert cfg.rewards["backflip_overspeed"].params["omega_max"] == OVERSPEED_OMEGA_MAX


def test_landing_rewards_gated_on_completion():
    # Every landing/standing reward is state-gated on flip completion (frontier
    # >= ~300°) — not a clock. "Do nothing" and a standing spawn earn nothing.
    cfg = make_microduck_backflip_env_cfg()
    for name in (
        "backflip_landing_composite",
        "backflip_upright_after_flip",
        "backflip_height_after_flip",
        "backflip_landing_sharp",
        "backflip_stand_tax",
    ):
        params = cfg.rewards[name].params
        assert params["gate_lo"] == LANDING_GATE_LO, f"{name} gate_lo"
        assert params["gate_hi"] == LANDING_GATE_HI, f"{name} gate_hi"
        assert params["gate_lo"] < params["gate_hi"]
    # The gate opens over the final ~55° before 2π.
    assert LANDING_GATE_LO == math.radians(300.0)
    assert LANDING_GATE_HI == math.radians(355.0)


def test_landing_rewards_use_standup_height():
    # Target height measured on THIS groundcontact model (don't carry across
    # models — the 5 mm-wrong STAND_Z lesson).
    cfg = make_microduck_backflip_env_cfg()
    assert STAND_Z == 0.115
    for name in (
        "backflip_landing_composite",
        "backflip_height_after_flip",
        "backflip_landing_sharp",
        "backflip_stand_tax",
    ):
        assert cfg.rewards[name].params["target_height"] == STAND_Z, name
    # The launch bootstrap cuts out just ABOVE the standing target.
    assert cfg.rewards["backflip_launch"].params["max_height"] == STAND_Z + 0.03


def test_landing_composite_targets_legs_at_home():
    cfg = make_microduck_backflip_env_cfg()
    params = cfg.rewards["backflip_landing_composite"].params
    assert params["joint_indices"] == _LEG_JOINTS
    # target_overrides=None -> the landing pose is HOME (default_joint_pos).
    assert params["target_overrides"] is None


# ── Airborne gate (the anti-cheat) ────────────────────────────────────────────


def test_airborne_gate_sensor_is_registered():
    """NAME IS LOAD-BEARING: _backflip_airborne reads exactly this sensor.

    The rotation accumulator is contact-INVERTED — it integrates only while NO
    robot geom touches the terrain — so a backward ground roll earns nothing and
    never opens the landing gate. If the sensor name drifts, the gate silently
    degrades to all-airborne and the policy farms a ground rock.
    """
    cfg = make_microduck_backflip_env_cfg()
    sensor_names = [s.name for s in cfg.scene.sensors]
    assert microduck_mdp._BACKFLIP_AIR_SENSOR in sensor_names
    assert microduck_mdp._BACKFLIP_AIR_SENSOR == "robot_ground_contact"
    # The other two sensors back the critic obs / self-collision reward.
    assert "feet_ground_contact" in sensor_names
    assert "self_collision" in sensor_names


def test_accumulator_sign_is_backward():
    # A backflip pitches nose-up/over-backward = NEGATIVE body-frame ω_y (forward
    # roll is +ω_y). The accumulator integrates -ω_y.
    assert microduck_mdp._BACKFLIP_ROT_SIGN == -1.0


# ── Spawn / reverse curriculum ────────────────────────────────────────────────


def test_set_backflip_state_runs_after_base_reset():
    # set_backflip_state overwrites the pose written by reset_base /
    # reset_robot_joints; events run in dict-insertion order, so it must come
    # AFTER (the mid-air tuck lerps FROM the HOME pose reset_robot_joints wrote).
    cfg = make_microduck_backflip_env_cfg()
    order = list(cfg.events.keys())
    assert "set_backflip_state" in order
    assert order.index("set_backflip_state") > order.index("reset_base")
    assert order.index("set_backflip_state") > order.index("reset_robot_joints")


def test_mid_air_spawns_are_genuinely_airborne():
    # Reverse-curriculum spawns must be in FREE FLIGHT (z above standing) so the
    # airborne accumulator keeps integrating from the first step; if they spawned
    # at/below standing height they'd touch ground and the gate would shut.
    cfg = make_microduck_backflip_env_cfg()
    params = cfg.events["set_backflip_state"].params
    assert params["air_z_min"] == BACKFLIP_AIR_Z_MIN
    assert params["air_z_min"] > params["standing_z_max"]
    # Backward angular momentum + a spin range that can finish the flip in airtime.
    assert params["air_omega_range"][1] > params["air_omega_range"][0] > 0.0


def test_spawn_mix_curriculum_shifts_to_standing():
    # Heavy mid-air early (the landing sub-task is learnable from day 0), shifting
    # toward standing starts as the full jump+flip is discovered. Mid-air never
    # goes to zero — it keeps the last mile practiced.
    cfg = make_microduck_backflip_env_cfg()
    assert "backflip_spawn_mix" in cfg.curriculum
    term = cfg.curriculum["backflip_spawn_mix"]
    assert term.params["event_name"] == "set_backflip_state"
    stages = term.params["param_stages"]

    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)

    standing = [s["params"]["standing_prob"] for s in stages]
    air = [s["params"]["air_prob"] for s in stages]
    assert standing == sorted(standing), "standing share must increase"
    assert air == sorted(air, reverse=True), "mid-air share must decrease"
    assert air[-1] > 0.0, "mid-air never goes to zero (last-mile practice)"
    for stage in stages:
        p = stage["params"]
        assert abs(p["standing_prob"] + p["air_prob"] - 1.0) < 1e-9


def test_spawn_mix_event_default_matches_stage_zero():
    # The curriculum manager runs BEFORE reset events, so the event's own default
    # is overwritten at the first reset; keep it consistent with stage 0 anyway.
    cfg = make_microduck_backflip_env_cfg()
    stage0 = cfg.curriculum["backflip_spawn_mix"].params["param_stages"][0]["params"]
    params = cfg.events["set_backflip_state"].params
    assert params["standing_prob"] == stage0["standing_prob"]
    assert params["air_prob"] == stage0["air_prob"]


def test_no_fall_termination():
    # Falling over IS the task (the robot is airborne and inverted mid-flip); a
    # bad_orientation termination would kill the episode on the first step.
    cfg = make_microduck_backflip_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_nan_state_watches_contact_sensor():
    # Hard landings can spike a contact force to non-finite a step before qpos
    # diverges; sensor_names lets the NaN guard catch it too.
    cfg = make_microduck_backflip_env_cfg()
    params = cfg.terminations["nan_state"].params
    assert params["sensor_names"] == ("feet_ground_contact",)


# ── Command & 61D obs parity (runtime hot-swap contract) ──────────────────────


def test_twist_command_is_neutralised():
    # No piloting: the flip deploys as a policy switch, the runtime leaves twist
    # at zero. Tiny non-zero sampling keeps the input neurons alive (AGENTS.md).
    cfg = make_microduck_backflip_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_twist_command_is_command_only():
    cfg = make_microduck_backflip_env_cfg()
    assert isinstance(cfg.commands["twist"], microduck_mdp.VelocityCommandCommandOnlyCfg)


def test_obs_parity_with_roulade_env():
    # 61D parity is mandatory: without it the ONNX won't load into a runtime slot.
    # The backflip mirrors the roulade's obs manipulation exactly.
    backflip = make_microduck_backflip_env_cfg()
    roulade = make_microduck_roulade_env_cfg()
    for grp in ("actor", "critic"):
        assert list(backflip.observations[grp].terms.keys()) == list(
            roulade.observations[grp].terms.keys()
        ), f"observation layout diverged on group {grp}"


def test_command_obs_slots_zero_padded():
    # head (4) and body (6) command slots are zero-padded — the flip uses no
    # commanded pose, but the slots stay for 61D family parity.
    cfg = make_microduck_backflip_env_cfg()
    for grp in ("actor", "critic"):
        head = cfg.observations[grp].terms["head_command"]
        body = cfg.observations[grp].terms["body_command"]
        assert head.func is microduck_mdp.zero_command_padding
        assert body.func is microduck_mdp.zero_command_padding
        assert head.params["dim"] == 4
        assert body.params["dim"] == 6


# ── DR / inherited sim2real stack ─────────────────────────────────────────────


def test_expand_bam_friction_fields_registered():
    # BAM invariant: every STANDALONE env cfg must register the friction-field
    # expansion startup event, else joint-friction DR is a silent no-op.
    cfg = make_microduck_backflip_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    assert cfg.events["expand_bam_friction_fields"].func is (
        microduck_mdp.expand_bam_friction_fields
    )
    assert "randomize_joint_friction" in cfg.events
    assert cfg.events["randomize_joint_friction"].func is (
        microduck_mdp.randomize_bam_friction
    )


def test_inherited_dr_events_survive():
    cfg = make_microduck_backflip_env_cfg()
    for name in (
        "randomize_com",
        "randomize_head_com",
        "randomize_armature",
        "randomize_joint_friction",
        "randomize_mass_inertia",
        "encoder_bias",
        "reset_action_history",
    ):
        assert name in cfg.events, f"DR/startup event lost: {name}"
    # A push mid-flip is incoherent — the interval push must be gone.
    assert "push_robot" not in cfg.events


def test_inherited_dr_curricula_survive():
    cfg = make_microduck_backflip_env_cfg()
    for name in ("com_range", "head_com_range"):
        assert name in cfg.curriculum, f"DR curriculum lost: {name}"


def test_late_polish_curricula_start_at_zero():
    # The standup timing lesson: any attempt-tax active during skill discovery
    # prevents the maneuver from being found at all. Settle polish ramps from 0.
    cfg = make_microduck_backflip_env_cfg()
    for name in ("arrival_damping_weight", "torque_rate_weight"):
        stages = cfg.curriculum[name].params["weight_stages"]
        assert stages[0]["weight"] == 0.0, f"{name} must start at zero"
        weights = [s["weight"] for s in stages]
        assert weights == sorted(weights, reverse=True), f"{name} must ramp negative"


def test_action_rate_ramp_stays_light():
    # Smoothness polish, kept light so it never blocks the fast flip (motion
    # blockers vs smoothness: action_rate damps jitter but must not gate the spin).
    cfg = make_microduck_backflip_env_cfg()
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    weights = [
        s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]
    ]
    assert all(w >= -0.5 for w in weights), "action_rate must stay light for a dynamic flip"


# ── Object identity & symmetry ────────────────────────────────────────────────


def test_trunk_asset_cfgs_are_distinct_objects():
    # mjlab resolves and MUTES SceneEntityCfg in place: an object shared between
    # terms gets stale indices. Each term must own its asset_cfg.
    cfg = make_microduck_backflip_env_cfg()
    names = (
        "backflip_launch",
        "backflip_landing_composite",
        "backflip_upright_after_flip",
        "backflip_height_after_flip",
        "backflip_landing_sharp",
        "backflip_stand_tax",
        "backflip_sagittal",
        "backflip_lateral_vel",
        "backflip_flatness",
        "arrival_damping",
    )
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg shared between reward terms"


def test_symmetry_enabled_for_sagittal_flip():
    # The backflip is sagittal / left-right symmetric; the mirror loss directly
    # fights the sideways-collapse (cartwheel) failure mode.
    assert MicroduckBackflipRlCfg.algorithm.symmetry_cfg is not None


def test_rl_cfg_bakes_normalizer_and_distinct_experiment():
    # Obs normalization is ON -> the normalizer must be baked into the ONNX by
    # export.py; distinct experiment_name so logs don't collide with siblings.
    assert MicroduckBackflipRlCfg.actor.obs_normalization is True
    assert MicroduckBackflipRlCfg.critic.obs_normalization is True
    assert MicroduckBackflipRlCfg.experiment_name == "microduck_backflip"
    assert MicroduckBackflipRlCfg.num_steps_per_env == 24
