"""OpenMANIPULATOR-X 해석적 IK 공통 모듈."""

import math

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ─── 링크 파라미터 ────────────────────────────────────────────────────────────

L1    = 0.0595
L2    = math.sqrt(0.024**2 + 0.128**2)
ALPHA = math.atan2(0.128, 0.024)
L3    = 0.124
L4    = 0.126

JOINT_LIMITS = [
    (-math.pi, math.pi),
    (-1.5,     1.5),
    (-1.5,     1.4),
    (-1.7,     1.97),
]

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']


def solve_ik(X: float, Y: float, Z: float):
    j1 = math.atan2(Y, X)
    r  = math.sqrt(X**2 + Y**2)

    dr = r - L4
    dz = Z - L1
    D  = math.sqrt(dr**2 + dz**2)

    if D > (L2 + L3) * 0.999:
        return None
    if D < abs(L2 - L3) * 1.001:
        return None

    c_psi = (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    c_psi = max(-1.0, min(1.0, c_psi))

    for psi in (-math.acos(c_psi), math.acos(c_psi)):
        s_psi  = math.sin(psi)
        gamma  = math.atan2(L3 * s_psi, L2 + L3 * c_psi)
        alpha1 = math.atan2(dz, dr) - gamma
        j2     = ALPHA - alpha1
        j3     = -psi - ALPHA
        j4     = -(j2 + j3)

        angles = [j1, j2, j3, j4]
        if all(lo <= a <= hi for a, (lo, hi) in zip(angles, JOINT_LIMITS)):
            return angles

    return None


def _shortest_path(target: float, current: float) -> float:
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints: list, current_joints: list,
                    speed: float, min_duration: float = 2.0):
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    duration = max(max_disp / speed, min_duration)

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    pt = JointTrajectoryPoint()
    pt.positions  = target_joints
    pt.velocities = [0.0] * 4
    secs  = int(duration)
    nsecs = int((duration - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points.append(pt)

    return traj, duration
