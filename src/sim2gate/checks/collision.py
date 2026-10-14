"""Lowest point of each robot body's authored collision geometry, for ground penetration (check 2).

Read once from the USD stage: for every body of the articulation (env 0), the collision shapes beneath that body's
prim (UsdPhysics.CollisionAPI on the shape or an ancestor below the body; instance proxies included; nested rigid
bodies excluded). Each shape becomes support points in the body's own frame plus a radius:

    Sphere    centre, radius r                 Capsule   the two hemisphere centres, radius r
    Cube      8 corners                        Cylinder  24 points on each end rim
    Mesh      its vertices (at most 512, evenly strided)

so that, for a body at world position p with rotation R, the lowest point of its geometry is
min_i (p + R q_i).z - r_i. Shapes are taken with their full transform relative to the body (offsets, rotations,
scale; radii are scaled by the largest axis scale, exact for uniform scale).

Isaac Lab 3.x reports body orientations as quaternions in (x, y, z, w) order; that is what lowest_z expects.
"""
import math

import torch

MAX_MESH_POINTS = 512
RIM_POINTS = 24


def _stride(points, cap):
    if len(points) <= cap:
        return points
    step = math.ceil(len(points) / cap)
    return points[::step]


def _shape_points(prim):
    """(local points [[x, y, z], ...], radius, kind) for a supported UsdGeom gprim, else None."""
    from pxr import UsdGeom
    t = prim.GetTypeName()
    if t == "Sphere":
        return [[0.0, 0.0, 0.0]], float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()), "sphere"
    if t in ("Capsule", "Cylinder"):
        g = UsdGeom.Capsule(prim) if t == "Capsule" else UsdGeom.Cylinder(prim)
        r, h = float(g.GetRadiusAttr().Get()), float(g.GetHeightAttr().Get())
        axis = "XYZ".index(str(g.GetAxisAttr().Get() or "Z"))
        if t == "Capsule":
            pts = []
            for s in (-0.5, 0.5):
                q = [0.0, 0.0, 0.0]; q[axis] = s * h; pts.append(q)
            return pts, r, "capsule"
        pts = []
        u, v = [i for i in range(3) if i != axis]
        for s in (-0.5, 0.5):
            for k in range(RIM_POINTS):
                a = 2 * math.pi * k / RIM_POINTS
                q = [0.0, 0.0, 0.0]; q[axis] = s * h; q[u] = r * math.cos(a); q[v] = r * math.sin(a); pts.append(q)
        return pts, 0.0, "cylinder"
    if t == "Cube":
        s = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
        return [[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)], 0.0, "box"
    if t == "Mesh":
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            return None
        return _stride([[float(p[0]), float(p[1]), float(p[2])] for p in pts], MAX_MESH_POINTS), 0.0, "mesh"
    return None


def _has_collision(prim, stop):
    """CollisionAPI on the prim or on an ancestor strictly below `stop` (the body prim) or on the body itself."""
    from pxr import UsdPhysics
    p = prim
    while p and p.IsValid():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if p.GetPath() == stop:
            return False
        p = p.GetParent()
    return False


def body_shapes(stage, body_prim):
    """Collision shapes under one body prim, in the body's frame: [{kind, points, radius, path}]."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = []
    it = iter(Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
    for prim in it:
        if prim != body_prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            it.PruneChildren()                       # another body's subtree
            continue
        if not _has_collision(prim, body_prim.GetPath()):
            continue
        sp = _shape_points(prim)
        if sp is None:
            continue
        pts, r, kind = sp
        m, _ = cache.ComputeRelativeTransform(prim, body_prim)
        scale = max(math.sqrt(sum(m[i][j] ** 2 for j in range(3))) for i in range(3))
        body_pts = []
        for q in pts:
            v = [q[0] * m[0][j] + q[1] * m[1][j] + q[2] * m[2][j] + m[3][j] for j in range(3)]  # row vectors
            body_pts.append(v)
        out.append({"kind": kind, "points": body_pts, "radius": r * scale, "path": str(prim.GetPath())})
    return out


def find_body_prim(stage, root_path, name):
    """The prim named `name` with RigidBodyAPI under root_path (instance proxies included), or None."""
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    hits = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
            if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    return hits[0] if hits else None


class CollisionGeometry:
    """Support points [B, P, 3] and radii [B, P] (NaN-padded) for the bodies of one robot, in body frames."""

    def __init__(self, shapes_by_body, device="cpu"):
        self.names = list(shapes_by_body)
        self.shapes = shapes_by_body
        P = max([sum(len(s["points"]) for s in v) for v in shapes_by_body.values()] + [1])
        B = len(self.names)
        pts = torch.full((B, P, 3), float("nan"), dtype=torch.float32)
        rad = torch.full((B, P), float("nan"), dtype=torch.float32)
        for b, n in enumerate(self.names):
            i = 0
            for s in shapes_by_body[n]:
                k = len(s["points"])
                pts[b, i:i + k] = torch.tensor(s["points"], dtype=torch.float32)
                rad[b, i:i + k] = s["radius"]
                i += k
        self.points, self.radii = pts.to(device), rad.to(device)
        self.has_shapes = torch.tensor([bool(shapes_by_body[n]) for n in self.names], device=device)

    @classmethod
    def from_stage(cls, stage, root_path, body_names, device="cpu"):
        shapes = {}
        for n in body_names:
            prim = find_body_prim(stage, root_path, n)
            shapes[n] = body_shapes(stage, prim) if prim is not None else []
        return cls(shapes, device)

    def summary(self):
        return {n: [{"kind": s["kind"], "radius_m": round(s["radius"], 6), "points": len(s["points"]),
                     "path": s["path"]} for s in v] for n, v in self.shapes.items()}

    def lowest_z(self, pos, quat_xyzw):
        """pos [N, B, 3], quat [N, B, 4] (x, y, z, w) -> lowest world z of each body's geometry [N, B]
        (NaN for a body without collision shapes)."""
        x, y, z, w = quat_xyzw.unbind(-1)
        r2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)  # R[2, :]
        pts = self.points.to(pos.dtype)
        zq = torch.einsum("nbk,bpk->nbp", r2, torch.nan_to_num(pts)) + pos[..., 2:3]
        zq = zq - torch.nan_to_num(self.radii.to(pos.dtype))
        zq = torch.where(torch.isnan(self.radii).unsqueeze(0), torch.full_like(zq, float("inf")), zq)
        low = zq.min(dim=-1).values
        return torch.where(self.has_shapes.unsqueeze(0), low, torch.full_like(low, float("nan")))
