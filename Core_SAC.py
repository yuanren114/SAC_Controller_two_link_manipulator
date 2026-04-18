import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from sac_agent import SAC, ReplayBuffer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class EpisodeDisturbance:
    def __init__(self, enabled=False, spike_step=None, spike_value=None):
        self.enabled = enabled
        self.spike_step = spike_step
        self.spike_value = np.zeros(2, dtype=np.float64) if spike_value is None else np.asarray(spike_value, dtype=np.float64)
        self.triggered = False

    def value_at(self, step_idx):
        if not self.enabled:
            return np.zeros(2, dtype=np.float64)
        if self.spike_step is None:
            return np.zeros(2, dtype=np.float64)
        if step_idx == self.spike_step and not self.triggered:
            self.triggered = True
            return self.spike_value.copy()
        return np.zeros(2, dtype=np.float64)


class TwoLink:
    def __init__(
        self,
        sim_mode="nominal",
        m1_nominal=2.0,
        m2_nominal=1.5,
        m1_actual=None,
        m2_actual=None,
        controller_bias_std=0.0,
    ):
        self.l1_draw = 180
        self.l2_draw = 140
        self.thickness = 16
        self.mode = sim_mode
        self.scale = 1000.0
        self.l1 = self.l1_draw / self.scale
        self.l2 = self.l2_draw / self.scale

        self.m1_nominal = float(m1_nominal)
        self.m2_nominal = float(m2_nominal)
        self.m1_actual = float(m1_nominal if m1_actual is None else m1_actual)
        self.m2_actual = float(m2_nominal if m2_actual is None else m2_actual)
        self.controller_bias_std = float(controller_bias_std)

        self.g = 9.81
        self.base = np.array([400.0, 500.0], dtype=np.float64)
        self.q = np.array([0.0, 0.0], dtype=np.float64)
        self.dq = np.array([0.0, 0.0], dtype=np.float64)
        self.qdd = np.array([0.0, 0.0], dtype=np.float64)

    def dynamics_terms(self, q, dq, use_nominal=False, rng=None):
        q1, q2 = q
        dq1, dq2 = dq

        if use_nominal:
            m1, m2 = self.m1_nominal, self.m2_nominal
        else:
            m1, m2 = self.m1_actual, self.m2_actual

        l1, l2 = self.l1, self.l2
        lc1, lc2 = l1 / 2, l2 / 2
        I1 = (1.0 / 12.0) * m1 * l1**2
        I2 = (1.0 / 12.0) * m2 * l2**2
        c2 = math.cos(q2)
        s2 = math.sin(q2)

        M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * c2)
        M12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
        M = np.array([[M11, M12], [M12, I2 + m2 * lc2**2]], dtype=np.float64)

        h = m2 * l1 * lc2 * s2
        C = np.array([
            -h * (2.0 * dq1 * dq2 + dq2**2),
            h * dq1**2,
        ], dtype=np.float64)

        G = np.array([
            (m1 * lc1 + m2 * l1) * self.g * math.cos(q1) + m2 * lc2 * self.g * math.cos(q1 + q2),
            m2 * lc2 * self.g * math.cos(q1 + q2),
        ], dtype=np.float64)

        if use_nominal and self.controller_bias_std > 0.0:
            local_rng = rng if rng is not None else np.random
            G = G + local_rng.normal(0.0, self.controller_bias_std, size=2)

        return M, C, G

    def step(self, tau, dt, disturbance=None):
        tau = np.clip(np.asarray(tau, dtype=np.float64), -5.0, 5.0)
        disturbance = np.zeros(2, dtype=np.float64) if disturbance is None else np.asarray(disturbance, dtype=np.float64)
        M, C, G = self.dynamics_terms(self.q, self.dq, use_nominal=False)
        qdd = np.linalg.solve(M, tau + disturbance - C - G - 0.5 * self.dq)
        if not np.all(np.isfinite(qdd)):
            qdd = np.zeros_like(self.dq)
        self.qdd = np.clip(qdd, -100.0, 100.0)
        self.dq = np.clip(self.dq + self.qdd * dt, -10.0, 10.0)
        if not np.all(np.isfinite(self.dq)):
            self.dq = np.zeros_like(self.dq)
        self.q = np.clip(self.q + self.dq * dt, -np.pi, np.pi)
        if not np.all(np.isfinite(self.q)):
            self.q = np.zeros_like(self.q)

    def fk(self):
        q1, q2 = self.q
        p0 = self.base
        p1 = p0 + np.array([self.l1_draw * math.cos(q1), self.l1_draw * math.sin(q1)], dtype=np.float64)
        p2 = p1 + np.array([self.l2_draw * math.cos(q1 + q2), self.l2_draw * math.sin(q1 + q2)], dtype=np.float64)
        return p0, p1, p2

    def ik4traject(self, x, y):
        r2 = x**2 + y**2
        c2 = np.clip((r2 - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2), -1.0, 1.0)
        q2 = math.atan2(math.sqrt(max(0.0, 1.0 - c2**2)), c2)
        q1 = math.atan2(y, x) - math.atan2(self.l2 * math.sin(q2), self.l1 + self.l2 * math.cos(q2))
        return np.array([q1, q2], dtype=np.float32)

    def pd_torque(self, q_target, dq_target, rng=None):
        Kp = np.array([40.0, 35.0], dtype=np.float64)
        Kd = np.array([10.0, 8.0], dtype=np.float64)
        _, _, G_nom = self.dynamics_terms(self.q, self.dq, use_nominal=True, rng=rng)
        tau = Kp * (q_target.astype(np.float64) - self.q) + Kd * (dq_target.astype(np.float64) - self.dq) + G_nom
        return np.clip(tau, -20.0, 20.0)

    def draw_link(self, screen, p_start, p_end, thickness, color):
        dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9 or not np.isfinite(length):
            return
        nx, ny, h = -dy / length, dx / length, thickness / 2
        pygame.draw.polygon(screen, color, [
            (p_start[0] + nx * h, p_start[1] + ny * h),
            (p_start[0] - nx * h, p_start[1] - ny * h),
            (p_end[0] - nx * h, p_end[1] - ny * h),
            (p_end[0] + nx * h, p_end[1] + ny * h),
        ])


@dataclass
class ExperimentConfig:
    command: str = "train"
    variant: str = "improved"
    control_mode: str = "direct_torque"
    trajectory_mode: str = "ellipse"
    seed: int = 7
    total_steps: int = 3000
    eval_interval: int = 5000
    eval_episodes: int = 5
    max_episode_steps: int = 3000
    batch_size: int = 256
    replay_size: int = 1000000
    start_steps: int = 5000
    update_after: int = 5000
    update_every: int = 1
    dt: float = 1 / 60

    radius_px: float = 60.0
    center_offset_px: float = 180.0
    ellipse_a_px: float = 120.0
    ellipse_b_px: float = 70.0
    ellipse_center_offset_px: float = 170.0
    omega: float = 0.6

    action_limit: float = 5.0
    checkpoint: str = ""
    deterministic_eval: bool = True

    m1_nominal: float = 2.0
    m2_nominal: float = 1.5
    train_m1_actual: float = 2.0
    train_m2_actual: float = 1.5
    test_m1_actual: float = 1.9
    test_m2_actual: float = 1.5
    controller_bias_std: float = 0.0

    train_use_disturbance: bool = False
    test_use_disturbance: bool = True
    disturbance_seed_train: int = 701
    disturbance_seed_test: int = 1701
    disturbance_min_step: int = 360
    disturbance_max_step: int = 1200
    disturbance_low: float = -5.0
    disturbance_high: float = 5.0

    time_render_compute_only: bool = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_dir(config):
    root = Path("logs")
    root.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{stamp}_{config.command}_{config.variant}_{config.control_mode}_seed{config.seed}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_config(config, path):
    with (path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)


def ik_for_point(arm, point_px):
    return arm.ik4traject((point_px[0] - arm.base[0]) / arm.scale, (point_px[1] - arm.base[1]) / arm.scale)


def pdf_target_m(t):
    return np.array([0.1 * np.sin(t) + 0.12, 0.1 * np.cos(t) + 0.12], dtype=np.float32)


def pdf_target_velocity_m(t):
    return np.array([0.1 * np.cos(t), -0.1 * np.sin(t)], dtype=np.float32)


def circle_target_m(arm, t, config):
    radius_m = config.radius_px / arm.scale
    center_m = np.array([0.0, -config.center_offset_px / arm.scale], dtype=np.float32)
    phase = config.omega * t
    return center_m + radius_m * np.array([np.cos(phase), np.sin(phase)], dtype=np.float32)


def circle_target_velocity_m(arm, t, config):
    radius_m = config.radius_px / arm.scale
    phase = config.omega * t
    return radius_m * config.omega * np.array([-np.sin(phase), np.cos(phase)], dtype=np.float32)


def ellipse_target_m(arm, t, config):
    a_m = config.ellipse_a_px / arm.scale
    b_m = config.ellipse_b_px / arm.scale
    cy_m = -config.ellipse_center_offset_px / arm.scale
    phase = config.omega * t
    return np.array([
        a_m * np.cos(phase),
        cy_m + b_m * np.sin(phase),
    ], dtype=np.float32)


def ellipse_target_velocity_m(arm, t, config):
    a_m = config.ellipse_a_px / arm.scale
    b_m = config.ellipse_b_px / arm.scale
    phase = config.omega * t
    return np.array([
        -a_m * config.omega * np.sin(phase),
        b_m * config.omega * np.cos(phase),
    ], dtype=np.float32)


def target_at_m(arm, t, config):
    if config.trajectory_mode == "ellipse":
        return ellipse_target_m(arm, t, config), ellipse_target_velocity_m(arm, t, config)
    if config.trajectory_mode == "circle":
        return circle_target_m(arm, t, config), circle_target_velocity_m(arm, t, config)
    if config.trajectory_mode == "pdf":
        return pdf_target_m(t), pdf_target_velocity_m(t)
    raise ValueError(f"Unknown trajectory_mode: {config.trajectory_mode}")


def point_m_to_px(arm, point_m):
    return (arm.base.astype(np.float32) + point_m.astype(np.float32) * arm.scale).astype(np.float32)


def ref_at(arm, t, dt, config):
    xd_m, dxd_m = target_at_m(arm, t, config)
    xd1_m, dxd1_m = target_at_m(arm, t + dt, config)
    xd2_m, _ = target_at_m(arm, t + 2 * dt, config)
    xd = point_m_to_px(arm, xd_m)
    xd1 = point_m_to_px(arm, xd1_m)
    xd2 = point_m_to_px(arm, xd2_m)
    q, q1, q2 = ik_for_point(arm, xd), ik_for_point(arm, xd1), ik_for_point(arm, xd2)
    dq = np.clip((q1 - q) / dt, -10.0, 10.0).astype(np.float32)
    dq1 = np.clip((q2 - q1) / dt, -10.0, 10.0).astype(np.float32)
    return {
        "xd": xd,
        "xd_next": xd1,
        "xd_m": xd_m,
        "xd_next_m": xd1_m,
        "dxd_m": dxd_m,
        "dxd_next_m": dxd1_m,
        "q_des": q,
        "q_des_next": q1,
        "dq_des": dq,
        "dq_des_next": dq1,
    }


def ee_px(arm):
    return arm.fk()[2].astype(np.float32)


def local_m(arm, point_px):
    return ((point_px.astype(np.float32) - arm.base.astype(np.float32)) / arm.scale).astype(np.float32)


def ee_velocity_m(arm):
    q1, q2 = arm.q
    dq1, dq2 = arm.dq
    s1, c1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    jac = np.array([
        [-arm.l1 * s1 - arm.l2 * s12, -arm.l2 * s12],
        [arm.l1 * c1 + arm.l2 * c12, arm.l2 * c12],
    ], dtype=np.float64)
    return (jac @ np.array([dq1, dq2], dtype=np.float64)).astype(np.float32)


def state_dim(config):
    return 16


def build_state(arm, ref, config, next_state=False):
    ee = ee_px(arm)
    q, dq = arm.q.astype(np.float32), arm.dq.astype(np.float32)
    x = local_m(arm, ee)
    dx = ee_velocity_m(arm)
    xd_m = ref["xd_next_m"] if next_state else ref["xd_m"]
    dxd_m = ref["dxd_next_m"] if next_state else ref["dxd_m"]
    e = x - xd_m
    de = dx - dxd_m
    return np.hstack([q, dq, x, dx, xd_m, dxd_m, e, de]).astype(np.float32)


def reset_arm(arm, config, t=None):
    if t is None:
        t = 0.0
    xd_m, _ = target_at_m(arm, t, config)
    start_px = point_m_to_px(arm, xd_m)
    q0 = ik_for_point(arm, start_px).astype(np.float64)
    xd1_m, _ = target_at_m(arm, t + config.dt, config)
    start1_px = point_m_to_px(arm, xd1_m)
    q1 = ik_for_point(arm, start1_px).astype(np.float64)
    dq0 = np.clip((q1 - q0) / config.dt, -10.0, 10.0)
    arm.q = q0
    arm.dq = dq0
    arm.qdd = np.zeros(2, dtype=np.float64)
    return t


def make_agent(config):
    limit = config.action_limit
    return SAC(
        state_dim=state_dim(config),
        action_dim=2,
        action_low=np.array([-limit, -limit], dtype=np.float32),
        action_high=np.array([limit, limit], dtype=np.float32),
    )


def build_arm(config, phase="train"):
    if phase == "train":
        m1_actual = config.train_m1_actual
        m2_actual = config.train_m2_actual
    else:
        m1_actual = config.test_m1_actual
        m2_actual = config.test_m2_actual
    return TwoLink(
        sim_mode=phase,
        m1_nominal=config.m1_nominal,
        m2_nominal=config.m2_nominal,
        m1_actual=m1_actual,
        m2_actual=m2_actual,
        controller_bias_std=config.controller_bias_std,
    )


def build_episode_disturbance(config, phase, episode_idx):
    use_disturbance = config.train_use_disturbance if phase == "train" else config.test_use_disturbance
    if not use_disturbance:
        return EpisodeDisturbance(enabled=False)

    seed_base = config.disturbance_seed_train if phase == "train" else config.disturbance_seed_test
    rng = np.random.default_rng(seed_base + int(episode_idx))
    hi = max(config.disturbance_min_step + 1, min(config.disturbance_max_step, config.max_episode_steps - 1))
    lo = min(config.disturbance_min_step, hi - 1)
    spike_step = int(rng.integers(lo, hi))
    spike_value = rng.uniform(config.disturbance_low, config.disturbance_high, size=2)
    return EpisodeDisturbance(enabled=True, spike_step=spike_step, spike_value=spike_value)


def control(arm, agent, state, ref, total_steps, config, deterministic=False):
    if total_steps < config.start_steps and not deterministic:
        action_env = np.random.uniform(-config.action_limit, config.action_limit, size=2).astype(np.float32)
        action_norm = (action_env / max(config.action_limit, 1e-6)).astype(np.float32)
    else:
        action_env, action_norm = agent.select_action(state, deterministic=deterministic)
        action_env, action_norm = action_env.astype(np.float32), action_norm.astype(np.float32)

    tau_pd = np.zeros(2, dtype=np.float32)
    tau_res = np.clip(action_env, -config.action_limit, config.action_limit).astype(np.float32)
    tau = tau_res.copy()
    return tau, tau_res, tau_pd, action_norm


def reward_metrics(arm, ref, prev_ee, action_norm, tau, tau_res, disturbance, config):
    ee = ee_px(arm)
    q, dq = arm.q.astype(np.float32), arm.dq.astype(np.float32)
    err_px = float(np.linalg.norm(ee - ref["xd_next"])) if np.all(np.isfinite(ee)) else 1e6
    err_m = err_px / arm.scale
    reward = 1.0 - math.exp(2.0 * (err_m - 0.02))
    dx = ee_velocity_m(arm)
    dxd = ref["dxd_next_m"].astype(np.float32)
    q_err = ref["q_des_next"].astype(np.float32) - q
    dq_err = dxd - dx
    metrics = {
        "reward": float(reward),
        "ee_error_px": err_px,
        "ee_error_m": err_m,
        "q_error_norm": float(np.linalg.norm(q_err)),
        "dq_error_norm": float(np.linalg.norm(dq_err)),
        "action_norm": float(np.linalg.norm(action_norm)),
        "action_saturation": float(np.mean(np.abs(action_norm) > 0.98)),
        "tau_total_norm": float(np.linalg.norm(tau)),
        "tau_residual_norm": float(np.linalg.norm(tau_res)),
        "disturbance_norm": float(np.linalg.norm(disturbance)),
        "q_abs_max": float(np.max(np.abs(q))),
        "dq_abs_max": float(np.max(np.abs(dq))),
        "success": float(err_m < 0.02),
    }
    return float(reward), metrics


def summarize(rows, total_reward, timing_stats=None, disturbance=None):
    errors = np.array([r["ee_error_m"] for r in rows], dtype=np.float64)
    acts = np.array([r["action_norm"] for r in rows], dtype=np.float64)
    sats = np.array([r["action_saturation"] for r in rows], dtype=np.float64)
    succ = np.array([r["success"] for r in rows], dtype=np.float64)
    out = {
        "episode_reward": float(total_reward),
        "episode_steps": len(rows),
        "mean_tracking_error_m": float(np.mean(errors)),
        "rms_tracking_error_m": float(np.sqrt(np.mean(errors**2))),
        "max_tracking_error_m": float(np.max(errors)),
        "final_tracking_error_m": float(errors[-1]),
        "success_rate": float(np.mean(succ)),
        "mean_action_norm": float(np.mean(acts)),
        "action_saturation_rate": float(np.mean(sats)),
        "final_q_abs_max": float(max(abs(rows[-1]["q1"]), abs(rows[-1]["q2"]))),
        "final_dq_abs_max": float(max(abs(rows[-1]["dq1"]), abs(rows[-1]["dq2"]))),
    }
    if timing_stats is not None:
        out.update(timing_stats)
    if disturbance is not None:
        out["disturbance_enabled"] = float(disturbance.enabled)
        out["disturbance_spike_step"] = -1 if disturbance.spike_step is None else int(disturbance.spike_step)
        out["disturbance_spike_tau1"] = float(disturbance.spike_value[0])
        out["disturbance_spike_tau2"] = float(disturbance.spike_value[1])
    return out


def append_csv(path, row, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted(row.keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_numeric(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def plot_trajectory(rows, path, title):
    if plt is None or not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["x_target_px"] for r in rows], [r["y_target_px"] for r in rows], label="target", linewidth=2)
    axes[0].plot([r["x_ee_px"] for r in rows], [r["y_ee_px"] for r in rows], label="actual", linewidth=2)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].invert_yaxis()
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot([r["ee_error_m"] for r in rows])
    axes[1].set_title("tracking error")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("m")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_training(path):
    if plt is None or not (Path(path) / "training_episodes.csv").exists():
        return
    rows = read_csv_numeric(Path(path) / "training_episodes.csv")
    if not rows:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot([r["episode"] for r in rows], [r["episode_reward"] for r in rows])
    axes[0].set_ylabel("episode reward")
    axes[0].grid(True)
    axes[1].plot([r["episode"] for r in rows], [r["mean_tracking_error_m"] for r in rows], label="mean")
    axes[1].plot([r["episode"] for r in rows], [r["final_tracking_error_m"] for r in rows], label="final")
    axes[1].set_ylabel("tracking error (m)")
    axes[1].set_xlabel("episode")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(Path(path) / "training_curves.png", dpi=140)
    plt.close(fig)


def plot_eval(csv_path, out_path):
    if plt is None or not Path(csv_path).exists():
        return
    rows = read_csv_numeric(csv_path)
    if not rows:
        return
    labels = [str(r.get("step_label", i)) for i, r in enumerate(rows)]
    y = [r["mean_tracking_error_m_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(rows)), y, marker="o")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean tracking error (m)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def aggregate_eval(rows):
    keys = [
        "episode_reward",
        "mean_tracking_error_m",
        "rms_tracking_error_m",
        "max_tracking_error_m",
        "final_tracking_error_m",
        "success_rate",
        "mean_action_norm",
        "action_saturation_rate",
        "mean_step_compute_time_ms",
        "mean_step_full_loop_time_ms",
    ]
    out = {"episodes": len(rows)}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def run_episode(agent, config, train=False, replay_buffer=None, update_state=None, deterministic=False, episode_idx=0, phase=None):
    phase = phase or ("train" if train else "test")
    arm = build_arm(config, phase=phase)
    disturbance = build_episode_disturbance(config, phase=phase, episode_idx=episode_idx)
    t = reset_arm(arm, config)
    rows, total_reward, last_info = [], 0.0, {}
    compute_step_times = []
    full_step_times = []

    for step in range(config.max_episode_steps):
        loop_t0 = time.perf_counter()
        ref = ref_at(arm, t, config.dt, config)
        state = build_state(arm, ref, config)
        prev_ee = ee_px(arm)
        total_seen = update_state["total_steps"] if update_state else 0

        compute_t0 = time.perf_counter()
        tau, tau_res, tau_pd, action_norm = control(arm, agent, state, ref, total_seen, config, deterministic)
        spike = disturbance.value_at(step)
        arm.step(tau, config.dt, disturbance=spike)
        next_ref = ref_at(arm, t, config.dt, config)
        next_state = build_state(arm, next_ref, config, next_state=True)
        reward, metrics = reward_metrics(arm, next_ref, prev_ee, action_norm, tau, tau_res, spike, config)
        compute_t1 = time.perf_counter()
        compute_step_times.append(compute_t1 - compute_t0)

        bad = not np.all(np.isfinite(next_state)) or np.any(np.abs(arm.q) > np.pi) or np.any(np.abs(arm.dq) > 20.0)
        done = step + 1 >= config.max_episode_steps or bad

        if train:
            replay_buffer.push(state, action_norm, reward, next_state, float(done))
            update_state["episode_reward"] += reward
            update_state["total_steps"] += 1
            if len(replay_buffer) >= config.batch_size and update_state["total_steps"] >= config.update_after:
                for _ in range(config.update_every):
                    last_info = agent.update(replay_buffer, config.batch_size)
                    update_state["last_info"] = last_info

        total_reward += reward
        ee = ee_px(arm)
        row = {
            "step": step,
            "t": t,
            "x_target_px": float(next_ref["xd_next"][0]),
            "y_target_px": float(next_ref["xd_next"][1]),
            "x_ee_px": float(ee[0]),
            "y_ee_px": float(ee[1]),
            "q1": float(arm.q[0]),
            "q2": float(arm.q[1]),
            "dq1": float(arm.dq[0]),
            "dq2": float(arm.dq[1]),
            "tau1": float(tau[0]),
            "tau2": float(tau[1]),
            "tau_pd1": float(tau_pd[0]),
            "tau_pd2": float(tau_pd[1]),
            "disturbance1": float(spike[0]),
            "disturbance2": float(spike[1]),
            "step_compute_time_ms": float((compute_t1 - compute_t0) * 1000.0),
            **metrics,
        }
        rows.append(row)
        t += config.dt
        loop_t1 = time.perf_counter()
        full_step_times.append(loop_t1 - loop_t0)

        if done:
            break

    timing_stats = {
        "mean_step_compute_time_ms": float(np.mean(compute_step_times) * 1000.0),
        "std_step_compute_time_ms": float(np.std(compute_step_times) * 1000.0),
        "mean_step_full_loop_time_ms": float(np.mean(full_step_times) * 1000.0),
        "std_step_full_loop_time_ms": float(np.std(full_step_times) * 1000.0),
    }
    out = summarize(rows, total_reward, timing_stats=timing_stats, disturbance=disturbance)
    out.update(last_info)
    return out, rows


def evaluate(agent, config, path, step_label="final", episodes=None):
    agent.eval_mode()
    eval_dir = Path(path) / "eval" / str(step_label)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for ep in range(episodes or config.eval_episodes):
        metrics, rows = run_episode(
            agent,
            config,
            train=False,
            deterministic=config.deterministic_eval,
            episode_idx=ep,
            phase="test",
        )
        metrics.update({"episode": ep, "step_label": step_label})
        metrics_rows.append(metrics)
        write_rows(eval_dir / f"episode_{ep:03d}_trajectory.csv", rows)
        plot_trajectory(rows, eval_dir / f"episode_{ep:03d}_trajectory.png", f"{step_label} episode {ep}")
    summary = aggregate_eval(metrics_rows)
    summary["step_label"] = step_label
    summary["test_m1_actual"] = float(config.test_m1_actual)
    summary["test_m2_actual"] = float(config.test_m2_actual)
    summary["test_use_disturbance"] = float(config.test_use_disturbance)
    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    append_csv(Path(path) / "evaluation.csv", summary)
    plot_eval(Path(path) / "evaluation.csv", Path(path) / "evaluation_tracking_error.png")
    agent.train_mode()
    return summary


def write_summary(path, config, final_eval, best_error, training_time_s=0.0):
    lines = [
        "# SAC Tracking Run Summary",
        "",
        f"- Command: `{config.command}`",
        f"- Variant: `{config.variant}`",
        f"- Control mode: `{config.control_mode}`",
        f"- Seed: `{config.seed}`",
        f"- Total training steps: `{config.total_steps}`",
        f"- Training time: `{training_time_s:.3f} s`",
        f"- Final mean tracking error: `{final_eval['mean_tracking_error_m_mean']:.6f} m`",
        f"- Final RMS tracking error: `{final_eval['rms_tracking_error_m_mean']:.6f} m`",
        f"- Final max tracking error: `{final_eval['max_tracking_error_m_mean']:.6f} m`",
        f"- Final success rate: `{final_eval['success_rate_mean']:.3f}`",
        f"- Mean test step time (compute only): `{final_eval['mean_step_compute_time_ms_mean']:.4f} ms`",
        f"- Mean test step time (full eval loop): `{final_eval['mean_step_full_loop_time_ms_mean']:.4f} ms`",
        f"- Best evaluation mean error: `{best_error:.6f} m`",
        f"- Train plant mass m1/m2: `{config.train_m1_actual:.3f}/{config.train_m2_actual:.3f}` kg",
        f"- Test plant mass m1/m2: `{config.test_m1_actual:.3f}/{config.test_m2_actual:.3f}` kg",
        f"- Train disturbance enabled: `{config.train_use_disturbance}`",
        f"- Test disturbance enabled: `{config.test_use_disturbance}`",
        "",
        "The active controller applies the SAC policy output directly as joint torque bounded by the action limit.",
    ]
    with (Path(path) / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def train(config):
    set_seed(config.seed)
    path = run_dir(config)
    save_config(config, path)
    agent = make_agent(config)
    if config.checkpoint:
        agent.load(config.checkpoint)
    buffer = ReplayBuffer(state_dim=state_dim(config), action_dim=2, capacity=config.replay_size)
    ckpt_dir = path / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    step_log = (path / "training_steps.jsonl").open("a", encoding="utf-8")
    state = {"total_steps": 0, "episode_reward": 0.0, "last_info": {}}

    train_wall_t0 = time.perf_counter()

    best = evaluate(agent, config, path, "step_0", episodes=max(1, min(3, config.eval_episodes)))["mean_tracking_error_m_mean"]
    agent.save(ckpt_dir / "best.pt")
    episode = 0
    while state["total_steps"] < config.total_steps:
        state["episode_reward"] = 0.0
        metrics, _ = run_episode(
            agent,
            config,
            train=True,
            replay_buffer=buffer,
            update_state=state,
            deterministic=False,
            episode_idx=episode,
            phase="train",
        )
        metrics.update({
            "episode": episode,
            "total_steps": state["total_steps"],
            "replay_size": len(buffer),
            "train_m1_actual": config.train_m1_actual,
            "train_m2_actual": config.train_m2_actual,
            "train_use_disturbance": float(config.train_use_disturbance),
        })
        metrics.update(state.get("last_info", {}))
        append_csv(path / "training_episodes.csv", metrics)
        step_log.write(json.dumps(metrics, sort_keys=True) + "\n")
        step_log.flush()

        if episode % 5 == 0:
            print(
                f"episode={episode} steps={state['total_steps']} reward={metrics['episode_reward']:.2f} "
                f"mean_err_m={metrics['mean_tracking_error_m']:.4f} rmse={metrics['rmse_tracking_error_m']:.4f} "
                f"train_step_ms={metrics['mean_step_compute_time_ms']:.4f}"
            )

        crossed_eval = (
            state["total_steps"] >= config.total_steps
            or state["total_steps"] // config.eval_interval > max(0, (state["total_steps"] - metrics["episode_steps"]) // config.eval_interval)
        )

        if crossed_eval:
            label = f"step_{state['total_steps']}"
            agent.save(ckpt_dir / f"{label}.pt")
            summary = evaluate(agent, config, path, label)

            print("\n===== EVAL REPORT =====")
            print(f"Step: {state['total_steps']}")
            print(f"Episodes: {summary['episodes']}")
            print(f"Mean tracking error:         {summary['mean_tracking_error_m_mean']:.6f} m")
            print(f"RMS tracking error:          {summary['rms_tracking_error_m_mean']:.6f} m")
            print(f"Max tracking error:          {summary['max_tracking_error_m_mean']:.6f} m")
            print(f"Final tracking error:        {summary['final_tracking_error_m_mean']:.6f} m")
            print(f"Episode reward:              {summary['episode_reward_mean']:.6f}")
            print(f"Success rate:                {summary['success_rate_mean']:.3f}")
            print(f"Mean action norm:            {summary['mean_action_norm_mean']:.6f}")
            print(f"Action sat. rate:            {summary['action_saturation_rate_mean']:.3f}")
            print(f"Mean test step time compute: {summary['mean_step_compute_time_ms_mean']:.6f} ms")
            print(f"Mean test step time full:    {summary['mean_step_full_loop_time_ms_mean']:.6f} ms")
            print(f"Current best error:          {best:.6f} m")

            if summary["mean_tracking_error_m_mean"] < best:
                best = summary["mean_tracking_error_m_mean"]
                agent.save(ckpt_dir / "best.pt")
                print(f"New best checkpoint saved: {best:.6f} m")

            print("=======================\n")
        episode += 1

    training_time_s = time.perf_counter() - train_wall_t0
    agent.save(ckpt_dir / "final.pt")
    final_eval = evaluate(agent, config, path, "final")
    with (Path(path) / "training_time.json").open("w", encoding="utf-8") as f:
        json.dump({"training_time_s": training_time_s}, f, indent=2)
    write_summary(path, config, final_eval, best, training_time_s=training_time_s)
    plot_training(path)
    step_log.close()
    print(f"Total training wall time: {training_time_s:.3f} s")
    print(f"Run saved to {path}")
    return path


def eval_checkpoint(config):
    set_seed(config.seed)
    path = run_dir(config)
    save_config(config, path)
    agent = make_agent(config)
    if config.checkpoint:
        agent.load(config.checkpoint)
    summary = evaluate(agent, config, path, "checkpoint", episodes=config.eval_episodes)
    write_summary(path, config, summary, summary["mean_tracking_error_m_mean"], training_time_s=0.0)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Evaluation saved to {path}")
    return path


def compare(config):
    set_seed(config.seed)
    path = run_dir(config)
    save_config(config, path)
    rows = []
    for label, variant, mode in [("pdf_direct_torque", "improved", "direct_torque")]:
        cfg = ExperimentConfig(**asdict(config))
        cfg.variant, cfg.control_mode = variant, mode
        agent = make_agent(cfg)
        if cfg.checkpoint:
            agent.load(cfg.checkpoint)
        summary = evaluate(agent, cfg, path / label, "compare", episodes=cfg.eval_episodes)
        row = {
            "label": label,
            "mean_tracking_error_m": summary["mean_tracking_error_m_mean"],
            "rmse_tracking_error_m": summary["rmse_tracking_error_m_mean"],
            "rms_tracking_error_m": summary["rms_tracking_error_m_mean"],
            "max_tracking_error_m": summary["max_tracking_error_m_mean"],
            "final_tracking_error_m": summary["final_tracking_error_m_mean"],
            "success_rate": summary["success_rate_mean"],
            "mean_step_compute_time_ms": summary["mean_step_compute_time_ms_mean"],
            "mean_step_full_loop_time_ms": summary["mean_step_full_loop_time_ms_mean"],
        }
        rows.append(row)
        append_csv(path / "comparison.csv", row, fields=list(row.keys()))
    write_report(path, rows)
    print(f"Comparison saved to {path}")
    return path


def write_report(path, rows):
    lines = [
        "# SAC Tracking Debug Report",
        "",
        "## Current Pipeline",
        "",
        "- `TwoLink` owns dynamics, inverse kinematics, forward kinematics, and pygame drawing.",
        "- Target generation supports ellipse, circle, and PDF task-space trajectories.",
        "- `sac_agent.py` contains the generic SAC actor, critics, replay buffer, updates, and checkpoint serialization.",
        "",
        "## Comparison Snapshot",
        "",
        f"- Mean tracking error: `{rows[0]['mean_tracking_error_m']:.6f} m`",
        f"- RMSE tracking error: `{rows[0]['rmse_tracking_error_m']:.6f} m`",
        f"- Max tracking error: `{rows[0]['max_tracking_error_m']:.6f} m`",
        f"- Success rate: `{rows[0]['success_rate']:.3f}`",
        f"- Mean step compute time: `{rows[0]['mean_step_compute_time_ms']:.4f} ms`",
        "",
        "## Remaining Work",
        "",
        "- Add more controller baselines if you want a true algorithm comparison table.",
        "- Keep train/test mismatch and disturbance settings identical across controllers for fair comparison.",
    ]
    with (Path(path) / "REPORT.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def render(config):
    set_seed(config.seed)
    agent = make_agent(config)
    deterministic = bool(config.checkpoint)
    if config.checkpoint:
        agent.load(config.checkpoint)

    pygame.init()
    screen = pygame.display.set_mode((1000, 700))
    pygame.display.set_caption(f"TwoLink SAC tracking - {config.variant}/{config.control_mode}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    arm = build_arm(config, phase="test")
    disturbance = build_episode_disturbance(config, phase="test", episode_idx=0)
    t = reset_arm(arm, config)
    ref_traj, ee_traj, total_steps = [], [], 0
    running = True
    last = {"reward": 0.0, "ee_error_m": 0.0}
    compute_times = []
    full_loop_times = []

    while running:
        loop_t0 = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ref = ref_at(arm, t, config.dt, config)
        state = build_state(arm, ref, config)
        prev_ee = ee_px(arm)

        compute_t0 = time.perf_counter()
        tau, tau_res, tau_pd, action_norm = control(arm, agent, state, ref, total_steps, config, deterministic)
        spike = disturbance.value_at(total_steps)
        arm.step(tau, config.dt, disturbance=spike)
        next_ref = ref_at(arm, t, config.dt, config)
        _, last = reward_metrics(arm, next_ref, prev_ee, action_norm, tau, tau_res, spike, config)
        compute_t1 = time.perf_counter()

        compute_times.append((compute_t1 - compute_t0) * 1000.0)
        ee = ee_px(arm)
        ref_traj.append(next_ref["xd_next"].copy())
        ee_traj.append(ee.copy())
        ref_traj, ee_traj = ref_traj[-1500:], ee_traj[-1500:]

        screen.fill((245, 245, 245))
        p0, p1, p2 = arm.fk()
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)):
            arm.draw_link(screen, p0, p1, arm.thickness, (60, 140, 220))
            arm.draw_link(screen, p1, p2, arm.thickness, (220, 120, 60))
            pygame.draw.circle(screen, (0, 0, 0), p0.astype(int), 8)
            pygame.draw.circle(screen, (120, 220, 20), p1.astype(int), 7)
            pygame.draw.circle(screen, (20, 180, 20), p2.astype(int), 6)
        if len(ref_traj) > 1:
            pygame.draw.lines(screen, (200, 50, 50), False, [p.astype(int) for p in ref_traj], 2)
        if len(ee_traj) > 1:
            pygame.draw.lines(screen, (50, 200, 50), False, [p.astype(int) for p in ee_traj], 2)
        pygame.draw.circle(screen, (220, 30, 30), next_ref["xd_next"].astype(int), 5)
        pygame.draw.circle(screen, (50, 200, 50), ee.astype(int), 5)

        mean_compute_ms = float(np.mean(compute_times)) if compute_times else 0.0
        texts = [
            f"steps={total_steps} mode={config.control_mode}",
            f"q=[{arm.q[0]:.2f}, {arm.q[1]:.2f}] dq=[{arm.dq[0]:.2f}, {arm.dq[1]:.2f}]",
            f"err={last['ee_error_m']:.4f} m reward={last['reward']:.2f}",
            f"tau=[{tau[0]:.2f}, {tau[1]:.2f}] disturb=[{spike[0]:.2f}, {spike[1]:.2f}]",
            f"avg step compute time={mean_compute_ms:.4f} ms",
            f"test m1/m2={config.test_m1_actual:.2f}/{config.test_m2_actual:.2f}",
        ]
        for i, text in enumerate(texts):
            screen.blit(font.render(text, True, (0, 0, 0)), (20, 20 + i * 25))

        pygame.display.flip()
        t += config.dt
        total_steps += 1
        clock.tick(60)
        loop_t1 = time.perf_counter()
        full_loop_times.append((loop_t1 - loop_t0) * 1000.0)

    if compute_times:
        print(f"Average render/test step compute time: {np.mean(compute_times):.6f} ms")
    if full_loop_times:
        print(f"Average render/test full loop time: {np.mean(full_loop_times):.6f} ms")
    pygame.quit()
    sys.exit()


def train_render(config):
    train_config = ExperimentConfig(**asdict(config))
    train_config.command = "train_render"
    trained_run = train(train_config)

    render_config = ExperimentConfig(**asdict(config))
    render_config.command = "render"
    render_config.checkpoint = str(Path(trained_run) / "checkpoints" / "best.pt")
    render(render_config)


def parse_args():
    parser = argparse.ArgumentParser(description="TwoLink SAC trajectory tracking")
    sub = parser.add_subparsers(dest="command", required=False)

    def common(p):
        p.add_argument("--variant", choices=["legacy", "improved"], default="improved")
        p.add_argument("--control-mode", choices=["direct_torque"], default="direct_torque")
        p.add_argument("--trajectory-mode", choices=["ellipse", "circle", "pdf"], default="ellipse")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--total-steps", type=int, default=24000)
        p.add_argument("--eval-interval", type=int, default=6000)
        p.add_argument("--eval-episodes", type=int, default=2)
        p.add_argument("--max-episode-steps", type=int, default=3000)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--start-steps", type=int, default=5000)
        p.add_argument("--update-after", type=int, default=5000)
        p.add_argument("--action-limit", type=float, default=5.0)
        p.add_argument("--checkpoint", default="")
        p.add_argument("--ellipse-a-px", type=float, default=120.0)
        p.add_argument("--ellipse-b-px", type=float, default=70.0)
        p.add_argument("--omega", type=float, default=0.6)
        p.add_argument("--radius-px", type=float, default=60.0)
        p.add_argument("--center-offset-px", type=float, default=180.0)
        p.add_argument("--ellipse-center-offset-px", type=float, default=170.0)

        p.add_argument("--m1-nominal", type=float, default=2.0)
        p.add_argument("--m2-nominal", type=float, default=1.5)
        p.add_argument("--train-m1-actual", type=float, default=2.0)
        p.add_argument("--train-m2-actual", type=float, default=1.5)
        p.add_argument("--test-m1-actual", type=float, default=1.9)
        p.add_argument("--test-m2-actual", type=float, default=1.5)
        p.add_argument("--controller-bias-std", type=float, default=0.0)

        p.add_argument("--train-use-disturbance", action="store_true")
        p.add_argument("--test-use-disturbance", action="store_true")
        p.add_argument("--disturbance-seed-train", type=int, default=701)
        p.add_argument("--disturbance-seed-test", type=int, default=1701)
        p.add_argument("--disturbance-min-step", type=int, default=360)
        p.add_argument("--disturbance-max-step", type=int, default=1200)
        p.add_argument("--disturbance-low", type=float, default=-5.0)
        p.add_argument("--disturbance-high", type=float, default=5.0)

    for name in ["train", "train-render", "eval", "render", "compare"]:
        common(sub.add_parser(name))
    return parser.parse_args()


def config_from_args(args):
    command = args.command or "render"
    return ExperimentConfig(
        command=command,
        variant=args.variant,
        control_mode=args.control_mode,
        trajectory_mode=args.trajectory_mode,
        seed=args.seed,
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        max_episode_steps=args.max_episode_steps,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        update_after=args.update_after,
        action_limit=args.action_limit,
        checkpoint=args.checkpoint,
        radius_px=args.radius_px,
        center_offset_px=args.center_offset_px,
        ellipse_a_px=args.ellipse_a_px,
        ellipse_b_px=args.ellipse_b_px,
        ellipse_center_offset_px=args.ellipse_center_offset_px,
        omega=args.omega,
        m1_nominal=args.m1_nominal,
        m2_nominal=args.m2_nominal,
        train_m1_actual=args.train_m1_actual,
        train_m2_actual=args.train_m2_actual,
        test_m1_actual=args.test_m1_actual,
        test_m2_actual=args.test_m2_actual,
        controller_bias_std=args.controller_bias_std,
        train_use_disturbance=args.train_use_disturbance,
        test_use_disturbance=args.test_use_disturbance,
        disturbance_seed_train=args.disturbance_seed_train,
        disturbance_seed_test=args.disturbance_seed_test,
        disturbance_min_step=args.disturbance_min_step,
        disturbance_max_step=args.disturbance_max_step,
        disturbance_low=args.disturbance_low,
        disturbance_high=args.disturbance_high,
    )


def main():
    config = config_from_args(parse_args())
    if config.command == "train":
        train(config)
    elif config.command == "train-render":
        train_render(config)
    elif config.command == "eval":
        eval_checkpoint(config)
    elif config.command == "compare":
        compare(config)
    else:
        render(config)


if __name__ == "__main__":
    main()
