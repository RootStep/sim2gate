"""sim2gate.checks.robot_params: the snapshot reads what the engine reports; compare lists every difference."""
import types

import torch

from sim2gate.checks import robot_params as rp
import test_exploits as te


def _env(armature=0.0, mass=6.9):
    d = types.SimpleNamespace(default_joint_pos=te._Buf(torch.zeros(2, 12)),
                              joint_armature=te._Buf(torch.full((2, 12), armature)),
                              body_mass=te._Buf(torch.full((2, 3), mass)), joint_stiffness=te._Buf(torch.zeros(2, 12)))
    robot = types.SimpleNamespace(data=d, joint_names=[f"j{i}" for i in range(12)], body_names=["base", "a", "b"],
                                  actuators={"legs": types.SimpleNamespace(stiffness=torch.full((2, 12), 25.0),
                                                                           damping=torch.full((2, 12), 0.5))})
    cfg = types.SimpleNamespace(sim=types.SimpleNamespace(dt=0.005, gravity=(0, 0, -9.81)), decimation=4)
    return types.SimpleNamespace(scene={"robot": robot}, cfg=cfg,
                                 sim=types.SimpleNamespace(physics_manager=types.SimpleNamespace(__name__="X")))


def test_same_robot_has_no_differences_and_missing_fields_are_listed():
    a, b = rp.snapshot(_env()), rp.snapshot(_env())
    assert rp.compare(a, b) == []
    assert "joint_damping" in a["missing"] and a["data"]["joint_armature"] == [0.0] * 12


def test_differences_are_found():
    a, b = rp.snapshot(_env()), rp.snapshot(_env(armature=0.02, mass=7.5))
    paths = {p for p, _, _ in rp.compare(a, b)}
    assert paths == {"data.joint_armature", "data.body_mass"}
