"""Checks 1 and 5 without Isaac Lab: the grid, the scoring, the nominal profile, the swept motor model and the
episode recorder, end to end on a fake environment whose walking breaks under known conditions."""
import json
import types

import pytest
import torch

from sim2gate.checks import isaaclab_conditions as ic
from sim2gate.checks import robustness as rb
from sim2gate.checks import transfer as tr
from sim2gate.checks.commands import make_command_set, pin
import test_exploits as te

EPS = 4                                    # episodes per group in these tests (64 in the plan)
CMDS = make_command_set(EPS, 20261012)


# ---- grid ------------------------------------------------------------------------------------------------------

def test_grid_matches_the_plan():
    g = rb.conditions()
    assert len(g) == 1 + 5 * 9 and g[0]["param"] is None
    for name, p in rb.PARAMS.items():
        v = rb.values(name)
        assert len(v) == 10 and v[0] == p["nominal"]
        assert v[6] == pytest.approx(p["edge"])                      # the plausible edge is the 7th value
        assert v[-1] == pytest.approx(p["nominal"] + 1.5 * (p["edge"] - p["nominal"]))
        others = [c for c in g if c["param"] == name]
        assert [c["index"] for c in others] == list(range(1, 10))
        for c in others:                                             # one parameter at a time
            assert {k for k, x in c["condition"].items() if x != rb.PARAMS[k]["nominal"]} == {name}
    assert rb.values("action_latency_ms") == [5.0 * i for i in range(10)]      # whole 5 ms physics steps
    assert rb.layout(g, 3)[:7] == [0, 0, 0, 1, 1, 1, 2]


def test_commands_tile_across_groups():
    env = types.SimpleNamespace(command_manager=te.FakeCommandManager(12))
    undo, info = pin(env, CMDS, tile=True)
    v = env.command_manager.get_term("base_velocity").vel_command_b
    assert info["tiled"] == 3 and torch.equal(v[0:4], v[4:8]) and torch.equal(v[0:4], v[8:12])
    with pytest.raises(ValueError):
        pin(types.SimpleNamespace(command_manager=te.FakeCommandManager(10)), CMDS, tile=True)
    with pytest.raises(ValueError):
        pin(types.SimpleNamespace(command_manager=te.FakeCommandManager(12)), CMDS)       # tiling is opt-in


# ---- fakes ------------------------------------------------------------------------------------------------------

class FakeDelayBuffer:
    """Isaac Lab's DelayBuffer semantics: per-env lag; if fewer entries since reset, the oldest available."""
    def __init__(self, history, n, device="cpu"):
        self.h, self.n = history, n
        self.lag = torch.zeros(n, dtype=torch.long)
        self.hist = [[] for _ in range(n)]

    def set_time_lag(self, lag, ids=None):
        self.lag[:] = lag
        if int(self.lag.max()) > self.h:
            raise ValueError("lag beyond history")

    def reset(self, ids=None):
        for i in (range(self.n) if ids is None else [int(x) for x in ids]):
            self.hist[i] = []

    def compute(self, data):
        out = torch.empty_like(data)
        for i in range(self.n):
            self.hist[i] = (self.hist[i] + [data[i].clone()])[-(self.h + 1):]
            k = min(int(self.lag[i]), len(self.hist[i]) - 1)
            out[i] = self.hist[i][-1 - k]
        return out


class FakeDCMotor:
    """The part of Isaac Lab's DCMotor the sweep relies on: PD effort, clipped, written back to the action."""
    def __init__(self, cfg, num_envs=1, device="cpu"):
        self.cfg, self._num_envs, self._device = cfg, num_envs, device

    def reset(self, env_ids):
        pass

    def compute(self, ca, jp, jv):
        self.computed_effort = 25.0 * (ca.joint_positions - jp) + 0.5 * (ca.joint_velocities - jv) + ca.joint_efforts
        self.applied_effort = self.computed_effort.clamp(-20, 20)
        ca.joint_efforts = self.applied_effort
        ca.joint_positions = ca.joint_velocities = None
        return ca


Motor = ic.swept_motor_class(FakeDCMotor, FakeDelayBuffer)


def _ca(n, pos):
    return types.SimpleNamespace(joint_positions=pos.clone(), joint_velocities=torch.zeros(n, 2),
                                 joint_efforts=torch.zeros(n, 2))


def test_motor_latency_delays_commands_per_env():
    m = Motor(None, num_envs=2)
    m.sim2gate_set(lag=torch.tensor([0, 2]))
    outs = [m.compute(_ca(2, torch.full((2, 2), float(t) * 0.01)), torch.zeros(2, 2), torch.zeros(2, 2)).joint_efforts
            for t in range(5)]
    assert [round(float(o[0, 0]), 4) for o in outs] == [0.0, 0.25, 0.5, 0.75, 1.0]       # no delay
    assert [round(float(o[1, 0]), 4) for o in outs] == [0.0, 0.0, 0.0, 0.25, 0.5]        # two physics steps late


def test_motor_strength_scales_the_produced_torque():
    m = Motor(None, num_envs=2)
    m.sim2gate_set(strength=torch.tensor([1.0, 0.85]))
    out = m.compute(_ca(2, torch.full((2, 2), 2.0)), torch.zeros(2, 2), torch.zeros(2, 2)).joint_efforts
    assert out[0, 0] == pytest.approx(20.0) and out[1, 0] == pytest.approx(17.0)       # clipped, then weaker
    assert torch.equal(m.applied_effort, out)


def _cfg(motor_cls=FakeDCMotor):
    ev = types.SimpleNamespace(
        add_base_mass=object(), base_com=object(), push_robot=None, base_external_force_torque=object(),
        reset_base=object(), physics_material=types.SimpleNamespace(params={
            "static_friction_range": (0.4, 1.2), "dynamic_friction_range": (0.3, 0.9), "restitution_range": (0, 0)}))

    class Cfg(types.SimpleNamespace):
        def to_dict(self):
            return {"decimation": self.decimation, "seed": self.seed, "events": sorted(vars(self.events)),
                    "sim": {"dt": self.sim.dt, "physics": str(self.sim.physics)}}
    return Cfg(events=ev, observations=types.SimpleNamespace(policy=types.SimpleNamespace(enable_corruption=False)),
               scene=types.SimpleNamespace(num_envs=7, robot=types.SimpleNamespace(
                   actuators={"base_legs": types.SimpleNamespace(class_type=motor_cls)})),
               sim=types.SimpleNamespace(dt=0.005, physics=types.SimpleNamespace(), use_newton_actuators=False),
               decimation=4, seed=42)


def test_nominal_profile():
    cfg = _cfg()
    rec = ic.apply_profile(cfg, motor_cls=Motor)
    assert rec["removed_events"] == ["add_base_mass", "base_com", "base_external_force_torque"]
    assert cfg.events.reset_base is not None                               # initial-state randomization kept
    assert cfg.events.physics_material.params["static_friction_range"] == (0.8, 0.8)
    assert cfg.events.physics_material.params["dynamic_friction_range"] == (0.6, 0.6)
    assert cfg.observations.policy.enable_corruption is True
    assert cfg.scene.robot.actuators["base_legs"].class_type is Motor
    other = _cfg(motor_cls=object)
    with pytest.raises(RuntimeError, match="DC motor"):
        ic.apply_profile(other, motor_cls=Motor)


def test_string_class_types_are_resolved():
    assert ic._resolve_class("json:JSONDecoder") is json.JSONDecoder


class FakeMaterial:
    def __init__(self, env, params):
        self.env = env

    def __call__(self, env, ids, static_friction_range, dynamic_friction_range, restitution_range, num_buckets,
                 asset_cfg, make_consistent=False):
        env.friction[ids] = static_friction_range[0]


class FakeMass:
    def __init__(self, env, params):
        pass

    def __call__(self, env, ids, asset_cfg, dist, op, distribution):
        assert op == "add"
        env.robot.data.body_mass.torch[ids, 0] += dist[0]


class FakeEntity:
    def __init__(self, name, body_names=None):
        self.name, self.body_names = name, body_names

    def resolve(self, scene):
        self.body_ids = [0]


API = types.SimpleNamespace(material=FakeMaterial, mass=FakeMass, entity=FakeEntity,
                            friction_readback=lambda robot: robot.env.friction)


class CondEnv:
    """A walking robot that breaks under known conditions: it falls at step 5 with >= 25 ms latency or >= 3.5 kg
    added; it tracks 0.3 m/s worse below 0.45 friction, 0.2 m/s worse below 0.8x strength or above 2.2x noise.
    Episodes are 20 steps."""
    T = 20

    def __init__(self, cfg, render_mode=None):
        self.cfg = cfg
        n = cfg.scene.num_envs
        self.num_envs, self.device, self.step_dt, self.max_episode_length = n, "cpu", 0.02, self.T
        self.built_with = {"corruption": cfg.observations.policy.enable_corruption,
                           "events": [k for k, v in vars(cfg.events).items() if v is not None]}
        motor_cls = cfg.scene.robot.actuators["base_legs"].class_type
        self.robot = types.SimpleNamespace(
            env=self, data=types.SimpleNamespace(root_lin_vel_b=te._Buf(torch.zeros(n, 3)),
                                                 body_mass=te._Buf(torch.full((n, 2), 6.9))),
            actuators={"base_legs": motor_cls(None, num_envs=n)})
        self.scene = {"robot": self.robot}
        self.friction = torch.full((n,), cfg.events.physics_material.params["static_friction_range"][0])
        self.command_manager = te.FakeCommandManager(n)
        on = cfg.observations.policy.enable_corruption          # Isaac Lab drops the noise when corruption is off
        noise = [types.SimpleNamespace(noise=types.SimpleNamespace(n_min=-0.1, n_max=0.1) if on else None),
                 types.SimpleNamespace(noise=types.SimpleNamespace(n_min=-1.5, n_max=1.5) if on else None)]
        self.observation_manager = types.SimpleNamespace(_group_obs_term_cfgs={"policy": noise},
                                                         _group_obs_term_names={"policy": ["base_lin_vel", "joint_vel"]})
        self.reward_manager = types.SimpleNamespace(compute=lambda dt=None: torch.zeros(n))
        self.sim = types.SimpleNamespace(physics_manager=types.SimpleNamespace(__name__="PhysxManager"))
        self.t = torch.zeros(n, dtype=torch.long)

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.t.zero_()
        term = self.command_manager.get_term("base_velocity")
        term._resample_command(torch.arange(self.num_envs))
        self.robot.actuators["base_legs"].reset(None)
        return torch.zeros(self.num_envs, 3), {}

    def step(self, actions):
        n = self.num_envs
        m = self.robot.actuators["base_legs"]
        added = self.robot.data.body_mass.torch[:, 0] - 6.9
        noise = self.observation_manager._group_obs_term_cfgs["policy"][0].noise.n_max
        noise = noise.flatten() / 0.1 if isinstance(noise, torch.Tensor) else torch.ones(n)
        err = torch.full((n,), 0.05)
        err += 0.3 * (self.friction < 0.45) + 0.2 * (m.sim2gate_strength.flatten() < 0.8) + 0.2 * (noise > 2.2)
        cmd = self.command_manager.get_term("base_velocity").vel_command_b
        v = cmd.clone()
        v[:, 0] -= err
        self.robot.data.root_lin_vel_b.torch[:] = v
        self.t += 1
        fall = (self.t == 5) & ((m.sim2gate_lag >= 5) | (added >= 3.5 - 1e-6))
        timeout = self.t >= self.T
        self.reward_manager.compute(dt=self.step_dt)
        dones = fall | timeout
        ids = dones.nonzero().flatten()
        self.t[ids] = 0
        if len(ids):
            self.command_manager.get_term("base_velocity")._resample_command(ids)
            v[ids] = 0.0                                            # a reset zeroes the state: read it pre-reset
            self.robot.data.root_lin_vel_b.torch[:] = v
        return torch.zeros(n, 3), torch.zeros(n), dones.long(), {"time_outs": timeout & ~fall}


class Wrapper:
    def __init__(self, env):
        self.unwrapped, self.num_envs = env, env.num_envs

    def reset(self):
        return self.unwrapped.reset()

    def step(self, actions):
        return self.unwrapped.step(actions)


class EnvCls(CondEnv):
    pass


def _play(tmp_path, mode="sweep", name="raw.json", ckpt=b"weights", play_args=None, physics="PhysxManager"):
    """Isaac Lab's play flow in miniature: build the env from its config, load the checkpoint, step until done."""
    play_args = play_args or ["--task", "T", "--checkpoint", "m.pt", "physics=isaacsim_physx",
                              "env.sim.use_newton_actuators=false"]
    rec = ic.ConditionRecorder(mode, str(tmp_path / name), CMDS, "cmds.json", play_args, episodes=EPS, api=API,
                               motor_cls=Motor)
    cls = type("E", (CondEnv,), {})
    undo = ic.install(rec, wrapper_cls=Wrapper, runner_cls=te.FakeRunner, env_cls=cls)
    try:
        ck = tmp_path / "model_999.pt"
        ck.write_bytes(ckpt)
        te.FakeRunner().load(str(ck))
        cfg = _cfg(motor_cls=FakeDCMotor)
        cfg.scene.num_envs = rec.num_envs
        env = cls(cfg)
        env.sim.physics_manager.__name__ = physics
        w = Wrapper(env)
        with pytest.raises(KeyboardInterrupt):
            for _ in range(CondEnv.T + 10):
                w.step(torch.zeros(env.num_envs, 12))
    finally:
        undo()
    return rec.raw, env


def test_profile_is_applied_before_the_env_is_built(tmp_path):
    raw, env = _play(tmp_path)
    assert env.built_with["corruption"] is True
    assert "add_base_mass" not in env.built_with["events"] and "base_com" not in env.built_with["events"]
    assert raw["profile"] == ic.PROFILE and raw["checkpoint_sha256"]


def test_sweep_records_every_first_episode_under_its_condition(tmp_path):
    raw, env = _play(tmp_path)
    eps = raw["episodes"]
    assert len(eps) == 46 * EPS and raw["all_ended"] and raw["steps_run"] == CondEnv.T
    rb_ = raw["conditions"]["readback"]
    g_lat = next(g for g in raw["groups"] if g["param"] == "action_latency_ms" and g["index"] == 5)
    assert rb_["action_latency_ms"][str(g_lat["group"])] == [25.0, 25.0]
    g_m = next(g for g in raw["groups"] if g["param"] == "added_base_mass_kg" and g["index"] == 3)
    assert rb_["added_base_mass_kg"][str(g_m["group"])] == [pytest.approx(1.5), pytest.approx(1.5)]
    g_f = next(g for g in raw["groups"] if g["param"] == "ground_friction" and g["index"] == 6)
    assert rb_["ground_friction"][str(g_f["group"])] == [pytest.approx(0.4), pytest.approx(0.4)]
    assert rb_["ground_friction"]["0"] == [pytest.approx(0.8), pytest.approx(0.8)]
    fell = {e["group"] for e in eps if e["fell"]}
    lat_groups = {g["group"] for g in raw["groups"] if g["param"] == "action_latency_ms" and g["value"] >= 25}
    mass_groups = {g["group"] for g in raw["groups"] if g["param"] == "added_base_mass_kg" and g["value"] >= 3.5}
    assert fell == lat_groups | mass_groups
    nom = [e for e in eps if e["group"] == 0]
    assert all(e["steps"] == CondEnv.T and e["mean_error_mps"] == pytest.approx(0.05) for e in nom)
    fallen = next(e for e in eps if e["fell"])
    assert fallen["steps"] == 5 and fallen["mean_error_mps"] == pytest.approx(0.05)   # read before the reset


def test_check1_finds_the_planted_cliffs(tmp_path):
    raw, _ = _play(tmp_path)
    succ = rb.success_reference([("P0 s1", raw)])
    assert succ["reference_error_mps"] == pytest.approx(0.05) and succ["threshold_mps"] == pytest.approx(0.075)
    res = rb.score(raw, succ)
    assert res["status"] == "scored" and res["nominal_success_rate"] == 1.0
    p = res["params"]
    assert p["action_latency_ms"]["cliff"] == 25.0 and p["action_latency_ms"]["flagged"]
    assert p["ground_friction"]["cliff"] == pytest.approx(0.4) and p["ground_friction"]["flagged"]
    assert p["added_base_mass_kg"]["cliff"] == 3.5 and not p["added_base_mass_kg"]["flagged"]      # beyond 3 kg
    assert p["motor_strength_scale"]["cliff"] == pytest.approx(0.775) and not p["motor_strength_scale"]["flagged"]
    assert p["observation_noise_scale"]["cliff"] == pytest.approx(2.333333, abs=1e-5)
    assert not p["observation_noise_scale"]["flagged"]
    assert res["flagged"] is True and {r["metric"] for r in res["risks"]} == {"action_latency_ms cliff",
                                                                              "ground_friction cliff"}
    lat = next(r for r in res["risks"] if r["metric"].startswith("action_latency"))
    assert lat["margin"] == pytest.approx(5.0) and lat["fix"]


def test_check1_is_invalid_when_the_policy_is_not_ready_or_the_setup_differs(tmp_path):
    raw, _ = _play(tmp_path)
    succ = rb.success_reference([("P0 s1", raw)])
    bad = json.loads(json.dumps(raw))
    for e in bad["episodes"]:
        if e["group"] == 0 and e["env"] % 2 == 0:
            e["fell"] = True
    res = rb.score(bad, succ)
    assert res["status"] == "invalid" and res["flagged"] is None and "policy not ready" in res["reasons"][0]
    other = json.loads(json.dumps(raw)); other["command_set_sha256"] = "x"
    assert rb.score(other, succ)["status"] == "invalid"
    other = json.loads(json.dumps(raw)); other["grid"]["sha256"] = "x"
    assert any("grid" in r for r in rb.score(other, succ)["reasons"])


def test_success_reference_refuses_falls_and_mixed_setups(tmp_path):
    raw, _ = _play(tmp_path)
    fell = json.loads(json.dumps(raw)); fell["episodes"][0]["fell"] = True
    with pytest.raises(rb.SuccessError, match="fell"):
        rb.success_reference([("a", raw), ("b", fell)])
    other = json.loads(json.dumps(raw)); other["profile"] = "other"
    with pytest.raises(rb.SuccessError, match="profile"):
        rb.success_reference([("a", raw), ("b", other)])


def test_latency_must_be_whole_physics_steps(tmp_path):
    rec = ic.ConditionRecorder("sweep", str(tmp_path / "r.json"), CMDS, "c", [], episodes=EPS, api=API,
                               motor_cls=Motor)
    cfg = _cfg(); cfg.scene.num_envs = rec.num_envs; cfg.sim.dt = 0.004
    rec.on_env_cfg(cfg)
    env = CondEnv(cfg)
    with pytest.raises(RuntimeError, match="whole physics steps"):
        rec._setup(Wrapper(env))


# ---- check 5 ----------------------------------------------------------------------------------------------------

def _pair(tmp_path, newton_slow=0.0, newton_fall=False, extra=None):
    px, _ = _play(tmp_path, "nominal", "px.json")
    nw = json.loads(json.dumps(px))
    nw["overrides"] = ["physics=newton_mjwarp", "env.sim.use_newton_actuators=false"] + (extra or [])
    nw["physics"]["manager"] = "NewtonMJWarpManager"
    nw["physics"]["solver_cfg"] = {"njmax": 65}
    nw["physics"]["versions"] = {"newton": "1.0"}
    for i, e in enumerate(nw["episodes"]):
        e["mean_error_mps"] += newton_slow
        e["fell"] = newton_fall and i % 2 == 0
    return px, nw


def test_check5_nominal_run_is_one_group(tmp_path):
    px, _ = _play(tmp_path, "nominal", "px.json")
    assert len(px["episodes"]) == EPS and px["groups"] == rb.conditions()[:1] and px["grid"] is None


def test_check5_passes_a_transferable_policy(tmp_path):
    px, nw = _pair(tmp_path, newton_slow=0.01)
    succ = rb.success_reference([("s1", px)])
    res = tr.score(px, nw, succ)
    assert res["parity"] == "passed" and res["status"] == "scored" and res["flagged"] is False
    assert res["metrics"]["error_ratio"] == pytest.approx(0.06 / 0.05)


@pytest.mark.parametrize("slow, fall, metric", [(0.06, False, "error_ratio"), (0.0, True, "newton_success_rate")])
def test_check5_flags_engine_overfit(tmp_path, slow, fall, metric):
    px, nw = _pair(tmp_path, newton_slow=slow, newton_fall=fall)
    succ = rb.success_reference([("s1", px)])
    res = tr.score(px, nw, succ)
    assert res["flagged"] is True and metric in {r["metric"] for r in res["risks"]}


@pytest.mark.parametrize("change", ["checkpoint", "commands", "tuned", "manager", "varied", "cfg"])
def test_check5_parity_failures_make_it_invalid(tmp_path, change):
    px, nw = _pair(tmp_path, extra=["env.sim.physics.solver_cfg.iterations=50"] if change == "tuned" else None)
    if change == "checkpoint":
        nw["checkpoint_sha256"] = "other"
    elif change == "commands":
        nw["command_set_sha256"] = "other"
    elif change == "manager":
        nw["physics"]["manager"] = "PhysxManager"
    elif change == "varied":
        nw["conditions"]["varied"] = ["ground_friction"]
    elif change == "cfg":
        nw["env_cfg_sha256"] = "other"
    succ = rb.success_reference([("s1", px)])
    res = tr.score(px, nw, succ)
    assert res["status"] == "invalid" and res["flagged"] is None and res["parity"] == "failed"


def test_check5_policy_not_ready_on_physx(tmp_path):
    px, nw = _pair(tmp_path)
    succ = rb.success_reference([("s1", px)])
    for e in px["episodes"]:
        e["fell"] = True
    res = tr.score(px, nw, succ)
    assert res["status"] == "invalid" and any("not ready" in r for r in res["reasons"])


def test_cfg_digest_ignores_physics_only():
    a = {"sim": {"dt": 0.005, "physics": "physx"}, "scene": {"robot": {"actuators": {"l": {"armature": 0.0}}}}, "x": 1}
    b = json.loads(json.dumps(a)); b["sim"]["physics"] = "newton"; b["scene"]["robot"]["actuators"]["l"]["armature"] = 0.02
    c = json.loads(json.dumps(a)); c["x"] = 2
    assert ic._digest(a) == ic._digest(b) != ic._digest(c)


def test_training_ranges_from_a_saved_env_yaml():
    pytest.importorskip("yaml")
    text = """events:
  physics_material:
    params:
      static_friction_range: [0.8, 0.8]
      asset_cfg:
        joint_ids: !!python/object/apply:builtins.slice [null, null, null]
  add_base_mass:
    params: {mass_distribution_params: [0.8, 1.25], operation: scale, distribution: log_uniform}
observations:
  policy: {enable_corruption: true}
"""
    r = ic.training_ranges(text)
    assert r["ground_friction"] == {"static_friction_range": [0.8, 0.8]}
    assert r["added_base_mass_kg"]["operation"] == "scale"
    assert r["action_latency_ms"] == "no delay in the actuator model"
