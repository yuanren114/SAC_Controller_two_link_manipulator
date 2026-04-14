import argparse
import csv
import json
import math
import random
import sys
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


class TwoLink:
    def __init__(self, sim_mode="Controllerbias"):
        self.l1_draw = 180
        self.l2_draw = 140
        self.thickness = 16
        self.mode = sim_mode
        self.scale = 1000.0
        self.l1 = self.l1_draw / self.scale
        self.l2 = self.l2_draw / self.scale
        self.m1 = 2.0
        self.m2 = 1.5
        self.g = 9.81
        self.base = np.array([400.0, 500.0], dtype=np.float64)
        self.q = np.array([0.0, 0.0], dtype=np.float64)
        self.dq = np.array([0.0, 0.0], dtype=np.float64)
        self.qdd = np.array([0.0, 0.0], dtype=np.float64)

    def dynamics_terms(self, q, dq):
        q1, q2 = q
        dq1, dq2 = dq
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        lc1, lc2 = l1 / 2, l2 / 2
        I1 = (1 / 12) * m1 * l1**2
        I2 = (1 / 12) * m2 * l2**2
        c2 = math.cos(q2)
        s2 = math.sin(q2)
        M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
        M12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
        M = np.array([[M11, M12], [M12, I2 + m2 * lc2**2]], dtype=np.float64)
        h = m2 * l1 * lc2 * s2
        C = np.array([-h * (2 * dq1 * dq2 + dq2**2), h * dq1**2], dtype=np.float64)
        G = np.array([
            (m1 * lc1 + m2 * l1) * self.g * math.cos(q1) + m2 * lc2 * self.g * math.cos(q1 + q2),
            m2 * lc2 * self.g * math.cos(q1 + q2),
        ], dtype=np.float64)
        return M, C, G

    def step(self, tau, dt):
        tau = np.clip(np.asarray(tau, dtype=np.float64), -20.0, 20.0)
        M, C, G = self.dynamics_terms(self.q, self.dq)
        qdd = np.linalg.solve(M, tau - C - G - 0.5 * self.dq)
        if not np.all(np.isfinite(qdd)):
            qdd = np.zeros_like(self.dq)
        self.qdd = np.clip(qdd, -100.0, 100.0)
        self.dq = np.clip(self.dq + self.qdd * dt, -10.0, 10.0)
        if not np.all(np.isfinite(self.dq)):
            self.dq = np.zeros_like(self.dq)
        self.q = np.clip(self.q + self.dq * dt, -3.0 * np.pi, 3.0 * np.pi)
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
        c2 = np.clip((r2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2 = math.atan2(math.sqrt(max(0.0, 1 - c2**2)), c2)
        q1 = math.atan2(y, x) - math.atan2(self.l2 * math.sin(q2), self.l1 + self.l2 * math.cos(q2))
        return np.array([q1, q2], dtype=np.float32)

    def pd_torque(self, q_target, dq_target):
        Kp = np.array([40.0, 35.0], dtype=np.float64)
        Kd = np.array([10.0, 8.0], dtype=np.float64)
        _, _, G = self.dynamics_terms(self.q, self.dq)
        tau = Kp * (q_target.astype(np.float64) - self.q) + Kd * (dq_target.astype(np.float64) - self.dq) + G
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
    control_mode: str = "residual_pd"
    seed: int = 7
    total_steps: int = 20000
    eval_interval: int = 5000
    eval_episodes: int = 5
    max_episode_steps: int = 600
    batch_size: int = 128
    replay_size: int = 100000
    start_steps: int = 1000
    update_after: int = 1000
    update_every: int = 1
    dt: float = 1.0 / 60.0
    radius_px: float = 60.0
    center_offset_px: float = 180.0
    omega: float = 0.5
    action_limit: float = 5.0
    checkpoint: str = ""
    deterministic_eval: bool = True


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


def ref_at(arm, t, dt, config):
    cx, cy, r, w = arm.base[0], arm.base[1] - config.center_offset_px, config.radius_px, config.omega
    xd = np.array([cx + r * np.cos(w * t), cy + r * np.sin(w * t)], dtype=np.float32)
    xd1 = np.array([cx + r * np.cos(w * (t + dt)), cy + r * np.sin(w * (t + dt))], dtype=np.float32)
    xd2 = np.array([cx + r * np.cos(w * (t + 2 * dt)), cy + r * np.sin(w * (t + 2 * dt))], dtype=np.float32)
    q, q1, q2 = ik_for_point(arm, xd), ik_for_point(arm, xd1), ik_for_point(arm, xd2)
    dq = np.clip((q1 - q) / dt, -10.0, 10.0).astype(np.float32)
    dq1 = np.clip((q2 - q1) / dt, -10.0, 10.0).astype(np.float32)
    return {"xd": xd, "xd_next": xd1, "q_des": q, "q_des_next": q1, "dq_des": dq, "dq_des_next": dq1}


def ee_px(arm):
    return arm.fk()[2].astype(np.float32)


def local_m(arm, point_px):
    return ((point_px.astype(np.float32) - arm.base.astype(np.float32)) / arm.scale).astype(np.float32)


def state_dim(config):
    return 12 if config.variant == "legacy" else 18


def build_state(arm, ref, config, next_state=False):
    ee = ee_px(arm)
    xd = ref["xd_next"] if next_state else ref["xd"]
    qd = ref["q_des_next"] if next_state else ref["q_des"]
    dqd = ref["dq_des_next"] if next_state else ref["dq_des"]
    if config.variant == "legacy":
        return np.hstack([arm.q.astype(np.float32), arm.dq.astype(np.float32), qd, dqd, ee, xd]).astype(np.float32)
    q, dq = arm.q.astype(np.float32), arm.dq.astype(np.float32)
    q_err, dq_err = qd - q, dqd - dq
    ee_err = local_m(arm, xd) - local_m(arm, ee)
    return np.hstack([
        np.sin(q), np.cos(q), dq / 10.0,
        np.sin(qd), np.cos(qd), dqd / 10.0,
        q_err / np.pi, dq_err / 10.0,
        ee_err / (arm.l1 + arm.l2),
    ]).astype(np.float32)


def reset_arm(arm, config, t=None):
    if t is None:
        t = np.random.uniform(0.0, 2.0 * np.pi / config.omega)
    ref = ref_at(arm, t, config.dt, config)
    arm.q = ref["q_des"].astype(np.float64)
    arm.dq = np.zeros(2, dtype=np.float64)
    return t


def make_agent(config):
    limit = 20.0 if config.control_mode == "direct_torque" else config.action_limit
    return SAC(
        state_dim=state_dim(config),
        action_dim=2,
        action_low=np.array([-limit, -limit], dtype=np.float32),
        action_high=np.array([limit, limit], dtype=np.float32),
    )


def control(arm, agent, state, ref, total_steps, config, deterministic=False):
    if total_steps < config.start_steps and not deterministic:
        action_env = np.random.uniform(-config.action_limit, config.action_limit, size=2).astype(np.float32)
        action_norm = (action_env / max(config.action_limit, 1e-6)).astype(np.float32)
    else:
        action_env, action_norm = agent.select_action(state, deterministic=deterministic)
        action_env, action_norm = action_env.astype(np.float32), action_norm.astype(np.float32)
    if config.control_mode == "residual_pd":
        tau_pd = arm.pd_torque(ref["q_des"], ref["dq_des"]).astype(np.float32)
        tau_res = np.clip(action_env, -config.action_limit, config.action_limit).astype(np.float32)
        tau = np.clip(tau_pd + tau_res, -20.0, 20.0).astype(np.float32)
    else:
        tau_pd = np.zeros(2, dtype=np.float32)
        tau_res = action_env.astype(np.float32)
        tau = np.clip(action_env, -20.0, 20.0).astype(np.float32)
    return tau, tau_res, tau_pd, action_norm


def reward_metrics(arm, ref, prev_ee, action_norm, tau, tau_res, config):
    ee = ee_px(arm)
    q, dq = arm.q.astype(np.float32), arm.dq.astype(np.float32)
    q_err = ref["q_des_next"].astype(np.float32) - q
    dq_err = ref["dq_des_next"].astype(np.float32) - dq
    err_px = float(np.linalg.norm(ee - ref["xd_next"])) if np.all(np.isfinite(ee)) else 1e6
    prev_err_px = float(np.linalg.norm(prev_ee - ref["xd"])) if np.all(np.isfinite(prev_ee)) else 1e6
    err_m = err_px / arm.scale
    progress_m = (prev_err_px - err_px) / arm.scale
    if config.variant == "legacy":
        reward = -1.5 * err_px - 0.01 * float(np.sum(dq**2)) - 0.05 * float(np.sum(action_norm**2))
        reward += 2.0 * (prev_err_px - err_px)
    else:
        reward = -40.0 * err_m + 20.0 * progress_m
        reward += -2.0 * float(np.sum(q_err**2))
        reward += -0.02 * float(np.sum(action_norm**2))
        reward += -0.005 * float(np.sum((tau_res / max(config.action_limit, 1e-6)) ** 2))
        reward += 1.0 if err_m < 0.015 else 0.0
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
        "q_abs_max": float(np.max(np.abs(q))),
        "dq_abs_max": float(np.max(np.abs(dq))),
        "success": float(err_m < 0.02),
    }
    return float(reward), metrics


def summarize(rows, total_reward):
    errors = np.array([r["ee_error_m"] for r in rows], dtype=np.float64)
    acts = np.array([r["action_norm"] for r in rows], dtype=np.float64)
    sats = np.array([r["action_saturation"] for r in rows], dtype=np.float64)
    succ = np.array([r["success"] for r in rows], dtype=np.float64)
    return {
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


def run_episode(agent, config, train=False, replay_buffer=None, update_state=None, deterministic=False):
    arm = TwoLink()
    t = reset_arm(arm, config)
    rows, total_reward, last_info = [], 0.0, {}
    for step in range(config.max_episode_steps):
        ref = ref_at(arm, t, config.dt, config)
        state = build_state(arm, ref, config)
        prev_ee = ee_px(arm)
        total_seen = update_state["total_steps"] if update_state else 0
        tau, tau_res, tau_pd, action_norm = control(arm, agent, state, ref, total_seen, config, deterministic)
        arm.step(tau, config.dt)
        next_ref = ref_at(arm, t, config.dt, config)
        next_state = build_state(arm, next_ref, config, next_state=True)
        reward, metrics = reward_metrics(arm, next_ref, prev_ee, action_norm, tau, tau_res, config)
        bad = not np.all(np.isfinite(next_state)) or np.any(np.abs(arm.q) > 10.0) or np.any(np.abs(arm.dq) > 20.0)
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
        rows.append({
            "step": step, "t": t,
            "x_target_px": float(next_ref["xd_next"][0]), "y_target_px": float(next_ref["xd_next"][1]),
            "x_ee_px": float(ee[0]), "y_ee_px": float(ee[1]),
            "q1": float(arm.q[0]), "q2": float(arm.q[1]),
            "dq1": float(arm.dq[0]), "dq2": float(arm.dq[1]),
            "tau1": float(tau[0]), "tau2": float(tau[1]),
            "tau_pd1": float(tau_pd[0]), "tau_pd2": float(tau_pd[1]),
            **metrics,
        })
        t += config.dt
        if done:
            break
    out = summarize(rows, total_reward)
    out.update(last_info)
    return out, rows


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
        "episode_reward", "mean_tracking_error_m", "rms_tracking_error_m",
        "max_tracking_error_m", "final_tracking_error_m", "success_rate",
        "mean_action_norm", "action_saturation_rate",
    ]
    out = {"episodes": len(rows)}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def evaluate(agent, config, path, step_label="final", episodes=None):
    agent.eval_mode()
    eval_dir = Path(path) / "eval" / str(step_label)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for ep in range(episodes or config.eval_episodes):
        metrics, rows = run_episode(agent, config, train=False, deterministic=config.deterministic_eval)
        metrics.update({"episode": ep, "step_label": step_label})
        metrics_rows.append(metrics)
        write_rows(eval_dir / f"episode_{ep:03d}_trajectory.csv", rows)
        plot_trajectory(rows, eval_dir / f"episode_{ep:03d}_trajectory.png", f"{step_label} episode {ep}")
    summary = aggregate_eval(metrics_rows)
    summary["step_label"] = step_label
    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    append_csv(Path(path) / "evaluation.csv", summary)
    plot_eval(Path(path) / "evaluation.csv", Path(path) / "evaluation_tracking_error.png")
    agent.train_mode()
    return summary


def write_summary(path, config, final_eval, best_error):
    lines = [
        "# SAC Tracking Run Summary", "",
        f"- Command: `{config.command}`",
        f"- Variant: `{config.variant}`",
        f"- Control mode: `{config.control_mode}`",
        f"- Seed: `{config.seed}`",
        f"- Total training steps: `{config.total_steps}`",
        f"- Final mean tracking error: `{final_eval['mean_tracking_error_m_mean']:.6f} m`",
        f"- Final RMS tracking error: `{final_eval['rms_tracking_error_m_mean']:.6f} m`",
        f"- Final max tracking error: `{final_eval['max_tracking_error_m_mean']:.6f} m`",
        f"- Final success rate: `{final_eval['success_rate_mean']:.3f}`",
        f"- Best evaluation mean error: `{best_error:.6f} m`", "",
        "The improved controller keeps the existing IK target and PD torque as the stabilizing baseline, while SAC learns bounded residual torque. This fixes the original mismatch where the code title described SAC plus PD but the loop applied SAC torque directly.",
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
    best = evaluate(agent, config, path, "step_0", episodes=max(1, min(3, config.eval_episodes)))["mean_tracking_error_m_mean"]
    agent.save(ckpt_dir / "best.pt")
    episode = 0
    while state["total_steps"] < config.total_steps:
        state["episode_reward"] = 0.0
        metrics, _ = run_episode(agent, config, train=True, replay_buffer=buffer, update_state=state)
        metrics.update({"episode": episode, "total_steps": state["total_steps"], "replay_size": len(buffer)})
        metrics.update(state.get("last_info", {}))
        append_csv(path / "training_episodes.csv", metrics)
        step_log.write(json.dumps(metrics, sort_keys=True) + "\n")
        step_log.flush()
        if episode % 5 == 0:
            print(f"episode={episode} steps={state['total_steps']} reward={metrics['episode_reward']:.2f} mean_err_m={metrics['mean_tracking_error_m']:.4f}")
        crossed_eval = state["total_steps"] >= config.total_steps or state["total_steps"] // config.eval_interval > max(0, (state["total_steps"] - metrics["episode_steps"]) // config.eval_interval)
        if crossed_eval:
            label = f"step_{state['total_steps']}"
            agent.save(ckpt_dir / f"{label}.pt")
            summary = evaluate(agent, config, path, label)
            if summary["mean_tracking_error_m_mean"] < best:
                best = summary["mean_tracking_error_m_mean"]
                agent.save(ckpt_dir / "best.pt")
        episode += 1
    agent.save(ckpt_dir / "final.pt")
    final_eval = evaluate(agent, config, path, "final")
    write_summary(path, config, final_eval, best)
    plot_training(path)
    step_log.close()
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
    write_summary(path, config, summary, summary["mean_tracking_error_m_mean"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Evaluation saved to {path}")
    return path


def compare(config):
    set_seed(config.seed)
    path = run_dir(config)
    save_config(config, path)
    rows = []
    for label, variant, mode in [("legacy_direct_torque", "legacy", "direct_torque"), ("improved_residual_pd", "improved", "residual_pd")]:
        cfg = ExperimentConfig(**asdict(config))
        cfg.variant, cfg.control_mode = variant, mode
        agent = make_agent(cfg)
        summary = evaluate(agent, cfg, path / label, "untrained", episodes=cfg.eval_episodes)
        row = {
            "label": label,
            "mean_tracking_error_m": summary["mean_tracking_error_m_mean"],
            "rms_tracking_error_m": summary["rms_tracking_error_m_mean"],
            "max_tracking_error_m": summary["max_tracking_error_m_mean"],
            "final_tracking_error_m": summary["final_tracking_error_m_mean"],
            "success_rate": summary["success_rate_mean"],
        }
        rows.append(row)
        append_csv(path / "comparison.csv", row, fields=list(row.keys()))
    write_report(path, rows)
    print(f"Comparison saved to {path}")
    return path


def write_report(path, rows):
    before, after = rows[0], rows[1]
    ratio = before["mean_tracking_error_m"] / max(after["mean_tracking_error_m"], 1e-9)
    lines = [
        "# SAC Tracking Debug Report", "",
        "## Current Pipeline", "",
        "- `TwoLink` in `Core_SAC.py` owns dynamics, inverse kinematics, forward kinematics, PD torque, and pygame drawing.",
        "- The previous control loop mixed target generation, SAC updates, reward calculation, checkpointing, and pygame rendering in one 60 Hz loop.",
        "- Circle targets are defined by radius, center offset, and angular rate; IK converts target end-effector points to joint targets.",
        "- `sac_agent.py` contains the generic SAC actor, critics, replay buffer, updates, and checkpoint serialization.", "",
        "## Likely Failure Points Found", "",
        "- The code described `SAC(delta_q) + PD`, but SAC output was applied directly as torque and `pd_torque()` was unused.",
        "- Training was coupled to pygame rendering and `clock.tick(60)`, slowing data collection.",
        "- Observations mixed radians, rad/s, and raw pixel coordinates without normalization.",
        "- The reward was dominated by pixel-scale tracking error and no metrics were persisted.",
        "- There was no deterministic evaluation entry point, best checkpoint, config snapshot, or trajectory plot.", "",
        "## Changes", "",
        "- Added `train`, `eval`, `render`, and `compare` subcommands.",
        "- Added timestamped `logs/` run folders with config snapshots, JSONL/CSV logs, checkpoints, summaries, and plots.",
        "- Added deterministic evaluation with per-episode trajectory CSV files and target-vs-actual plots.",
        "- Added normalized observations and meter-scale reward for the improved variant.",
        "- Added `residual_pd` mode so SAC learns bounded residual torque on top of the existing PD baseline.", "",
        "## Controlled Comparison", "",
        f"- Legacy direct-torque untrained mean error: `{before['mean_tracking_error_m']:.6f} m`",
        f"- Improved residual-PD untrained mean error: `{after['mean_tracking_error_m']:.6f} m`",
        f"- Mean-error ratio legacy/improved: `{ratio:.2f}x`",
        f"- Legacy success rate: `{before['success_rate']:.3f}`",
        f"- Improved success rate: `{after['success_rate']:.3f}`", "",
        "## Remaining Work", "",
        "- Longer training is still needed before claiming convergence of the residual SAC policy.",
        "- Next tuning should focus on residual action limits, reward weights, and target-velocity tracking.",
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
    arm = TwoLink()
    t = reset_arm(arm, config)
    ref_traj, ee_traj, total_steps = [], [], 0
    running = True
    last = {"reward": 0.0, "ee_error_m": 0.0}
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        ref = ref_at(arm, t, config.dt, config)
        state = build_state(arm, ref, config)
        prev_ee = ee_px(arm)
        tau, tau_res, tau_pd, action_norm = control(arm, agent, state, ref, total_steps, config, deterministic)
        arm.step(tau, config.dt)
        next_ref = ref_at(arm, t, config.dt, config)
        _, last = reward_metrics(arm, next_ref, prev_ee, action_norm, tau, tau_res, config)
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
        texts = [
            f"steps={total_steps} mode={config.control_mode}",
            f"q=[{arm.q[0]:.2f}, {arm.q[1]:.2f}] dq=[{arm.dq[0]:.2f}, {arm.dq[1]:.2f}]",
            f"err={last['ee_error_m']:.4f} m reward={last['reward']:.2f}",
            f"tau=[{tau[0]:.2f}, {tau[1]:.2f}] residual=[{tau_res[0]:.2f}, {tau_res[1]:.2f}]",
        ]
        for i, text in enumerate(texts):
            screen.blit(font.render(text, True, (0, 0, 0)), (20, 20 + i * 25))
        pygame.display.flip()
        t += config.dt
        total_steps += 1
        clock.tick(60)
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
        p.add_argument("--control-mode", choices=["direct_torque", "residual_pd"], default="residual_pd")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--total-steps", type=int, default=20000)
        p.add_argument("--eval-interval", type=int, default=5000)
        p.add_argument("--eval-episodes", type=int, default=5)
        p.add_argument("--max-episode-steps", type=int, default=600)
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--start-steps", type=int, default=1000)
        p.add_argument("--update-after", type=int, default=1000)
        p.add_argument("--action-limit", type=float, default=5.0)
        p.add_argument("--checkpoint", default="")

    for name in ["train", "train-render", "eval", "render", "compare"]:
        common(sub.add_parser(name))
    return parser.parse_args()


def config_from_args(args):
    command = args.command or "render"
    return ExperimentConfig(
        command=command,
        variant=args.variant,
        control_mode=args.control_mode,
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
