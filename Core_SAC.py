import math
import sys
import os
import numpy as np
import pygame
import random

from sac_agent import SAC, ReplayBuffer


class TwoLink:
    def __init__(self, sim_mode='Controllerbias'):
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
        M21 = M12
        M22 = I2 + m2 * lc2**2

        M = np.array([[M11, M12],
                      [M21, M22]], dtype=np.float64)

        h = m2 * l1 * lc2 * s2

        C = np.array([
            -h * (2 * dq1 * dq2 + dq2**2),
             h * (dq1**2)
        ], dtype=np.float64)

        G = np.array([
            (m1 * lc1 + m2 * l1) * self.g * math.cos(q1) + m2 * lc2 * self.g * math.cos(q1 + q2),
            m2 * lc2 * self.g * math.cos(q1 + q2)
        ], dtype=np.float64)

        # if self.mode == 'Controllerbias':
        #     G += random.uniform(-0.05, 0.05)

        return M, C, G

    def step(self, tau, dt):
        tau = np.asarray(tau, dtype=np.float64)
        tau = np.clip(tau, -20.0, 20.0)

        M, C, G = self.dynamics_terms(self.q, self.dq)

        damping = 0.5 * self.dq
        rhs = tau - C - G - damping

        qdd = np.linalg.solve(M, rhs)

        if not np.all(np.isfinite(qdd)):
            qdd = np.zeros_like(self.dq)

        qdd = np.clip(qdd, -100.0, 100.0)
        self.qdd = qdd.copy()

        self.dq += qdd * dt
        if not np.all(np.isfinite(self.dq)):
            self.dq = np.zeros_like(self.dq)
        self.dq = np.clip(self.dq, -10.0, 10.0)

        self.q += self.dq * dt
        if not np.all(np.isfinite(self.q)):
            self.q = np.zeros_like(self.q)

        self.q = np.clip(self.q, -3.0 * np.pi, 3.0 * np.pi)

    def fk(self):
        q1, q2 = self.q

        p0 = self.base
        p1 = p0 + np.array([
            self.l1_draw * math.cos(q1),
            self.l1_draw * math.sin(q1)
        ], dtype=np.float64)
        p2 = p1 + np.array([
            self.l2_draw * math.cos(q1 + q2),
            self.l2_draw * math.sin(q1 + q2)
        ], dtype=np.float64)

        return p0, p1, p2

    def ik4traject(self, x, y):
        r2 = x**2 + y**2
        c2 = (r2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        c2 = np.clip(c2, -1.0, 1.0)

        s2 = math.sqrt(max(0.0, 1 - c2**2))
        q2 = math.atan2(s2, c2)

        subx = self.l1 + self.l2 * math.cos(q2)
        suby = self.l2 * math.sin(q2)
        q1 = math.atan2(y, x) - math.atan2(suby, subx)
        return np.array([q1, q2], dtype=np.float32)

    def pd_torque(self, q_target, dq_target):
        Kp = np.array([40.0, 35.0], dtype=np.float64)
        Kd = np.array([10.0, 8.0], dtype=np.float64)

        q_err = q_target.astype(np.float64) - self.q
        dq_err = dq_target.astype(np.float64) - self.dq

        _, _, G = self.dynamics_terms(self.q, self.dq)

        tau = Kp * q_err + Kd * dq_err + G
        tau = np.clip(tau, -20.0, 20.0)
        return tau

    def control(self, agent, state, deterministic=False):
        tau_env, tau_norm = agent.select_action(state, deterministic=deterministic)
        tau_env = np.clip(tau_env, -20.0, 20.0)
        return tau_env.astype(np.float32), tau_norm.astype(np.float32)

    def draw_link(self, screen, p_start, p_end, thickness, color):
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        L = math.hypot(dx, dy)
        if L < 1e-9 or not np.isfinite(L):
            return

        nx = -dy / L
        ny = dx / L
        h = thickness / 2

        corners = [
            (p_start[0] + nx * h, p_start[1] + ny * h),
            (p_start[0] - nx * h, p_start[1] - ny * h),
            (p_end[0]   - nx * h, p_end[1]   - ny * h),
            (p_end[0]   + nx * h, p_end[1]   + ny * h),
        ]
        pygame.draw.polygon(screen, color, corners)


def main():
    checkpoint_path = "sac_pd_qtarget.pt"

    ee_traj = []
    ref_traj = []
    MAX_TRAJ = 1500

    pygame.init()
    screen = pygame.display.set_mode((1000, 700))
    pygame.display.set_caption("TwoLink + SAC(delta_q) + PD")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    arm = TwoLink()

    arm.q = np.random.uniform(-np.pi, np.pi, size=(2,)).astype(np.float64)
    arm.dq = np.random.uniform(-0.2, 0.2, size=(2,)).astype(np.float64)

    # state = [q, dq, q_des, dq_des, ee, xd]
    state_dim = 12
    action_dim = 2

    # SAC action is delta_q, not torque now
    # so action bounds are in joint offset units
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=np.array([-20.0, -20.0], dtype=np.float32),
        action_high=np.array([20.0, 20.0], dtype=np.float32),
    )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=100000,
    )

    # if os.path.exists(checkpoint_path):
    #     print(f"Loading checkpoint from {checkpoint_path}")
    #     agent.load(checkpoint_path)

    batch_size = 64
    start_steps = 2000
    update_after = 1000
    update_every = 1
    max_episode_steps = 1000

    r = 60.0
    d = 180.0
    omega = 0.5

    cx = arm.base[0]
    cy = arm.base[1] - d

    total_steps = 0
    episode_steps = 0
    episode_return = 0.0
    episode_idx = 0
    last_info = {}

    # initialize the arm on the reference circle using IK
    t = np.random.uniform(0.0, 2.0 * np.pi / omega)
    init_dt = 1.0 / 60.0

    xd_init = np.array([
        cx + r * np.cos(omega * t),
        cy + r * np.sin(omega * t),
    ], dtype=np.float32)

    xd_init_next = np.array([
        cx + r * np.cos(omega * (t + init_dt)),
        cy + r * np.sin(omega * (t + init_dt)),
    ], dtype=np.float32)

    x_local_init = (xd_init[0] - arm.base[0]) / arm.scale
    y_local_init = (xd_init[1] - arm.base[1]) / arm.scale

    x_local_init_next = (xd_init_next[0] - arm.base[0]) / arm.scale
    y_local_init_next = (xd_init_next[1] - arm.base[1]) / arm.scale

    q_init = arm.ik4traject(x_local_init, y_local_init)
    q_init_next = arm.ik4traject(x_local_init_next, y_local_init_next)

    dq_init = (q_init_next - q_init) / init_dt
    dq_init = np.clip(dq_init, -10.0, 10.0).astype(np.float64)

    arm.q = q_init.astype(np.float64)
    arm.dq = np.zeros(2, dtype=np.float64)

    running = True

    running = True
    while running:
        dt = 1.0 / 60.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        xd = np.array([
            cx + r * np.cos(omega * t),
            cy + r * np.sin(omega * t),
        ], dtype=np.float32)

        xd_next = np.array([
            cx + r * np.cos(omega * (t + dt)),
            cy + r * np.sin(omega * (t + dt)),
        ], dtype=np.float32)

        ref_traj.append(xd.copy())
        if len(ref_traj) > MAX_TRAJ:
            ref_traj.pop(0)

        # IK reference
        x_local = (xd[0] - arm.base[0]) / arm.scale
        y_local = (xd[1] - arm.base[1]) / arm.scale

        x_local_next = (xd_next[0] - arm.base[0]) / arm.scale
        y_local_next = (xd_next[1] - arm.base[1]) / arm.scale

        q_des = arm.ik4traject(x_local, y_local)
        q_des_next = arm.ik4traject(x_local_next, y_local_next)

        dq_des = (q_des_next - q_des) / dt
        dq_des = np.clip(dq_des, -10.0, 10.0).astype(np.float32)

        p0, p1, p2 = arm.fk()
        ee = p2.astype(np.float32)

        state = np.hstack([
            arm.q.astype(np.float32),
            arm.dq.astype(np.float32),
            q_des.astype(np.float32),
            dq_des.astype(np.float32),
            ee,
            xd
        ]).astype(np.float32)

        if total_steps < start_steps:
            tau_env = np.random.uniform(-2.0, 2.0, size=(2,)).astype(np.float32)
            tau_norm = (tau_env / 20.0).astype(np.float32)
        else:
            tau_env, tau_norm = arm.control(
                agent,
                state,
                deterministic=False,
            )

        arm.step(tau_env, dt)

        xd_2next = np.array([
            cx + r * np.cos(omega * (t + 2.0 * dt)),
            cy + r * np.sin(omega * (t + 2.0 * dt)),
        ], dtype=np.float32)

        x_local_2next = (xd_2next[0] - arm.base[0]) / arm.scale
        y_local_2next = (xd_2next[1] - arm.base[1]) / arm.scale

        q_des_2next = arm.ik4traject(x_local_2next, y_local_2next)
        dq_des_next = (q_des_2next - q_des_next) / dt
        dq_des_next = np.clip(dq_des_next, -10.0, 10.0).astype(np.float32)

        p0, p1, p2 = arm.fk()
        ee_next = p2.astype(np.float32)

        next_state = np.hstack([
            arm.q.astype(np.float32),
            arm.dq.astype(np.float32),
            q_des_next.astype(np.float32),
            dq_des_next.astype(np.float32),
            ee_next,
            xd_next
        ]).astype(np.float32)

        if np.all(np.isfinite(ee_next)):
            ee_traj.append(ee_next.copy())
            if len(ee_traj) > MAX_TRAJ:
                ee_traj.pop(0)

        q_now = arm.q.astype(np.float32)
        dq_now = arm.dq.astype(np.float32)

        q_err = q_des - q_now
        dq_err = dq_des - dq_now
        ee_err = np.linalg.norm(ee_next - xd_next) if np.all(np.isfinite(ee_next)) else 1e6

        ee_err_prev = np.linalg.norm(ee - xd) if np.all(np.isfinite(ee)) else 1e6
        progress_reward = 2.0 * (ee_err_prev - ee_err)


        u_penalty = np.sum((tau_env / 20.0) ** 2)
        dq_penalty = np.sum(dq_now ** 2)

        reward = (
            -1.5 * ee_err
            -0.01 * dq_penalty
            -0.05 * u_penalty
            + 2.0 * (ee_err_prev - ee_err)
        )

        episode_steps += 1

        bad_state = (
            not np.all(np.isfinite(next_state)) or
            np.any(np.abs(arm.q) > 10.0) or
            np.any(np.abs(arm.dq) > 20.0)
        )

        done = (episode_steps >= max_episode_steps) or bad_state

        replay_buffer.push(
            state=state,
            action=tau_norm,
            reward=float(reward),
            next_state=next_state,
            done=float(done),
        )

        episode_return += reward
        total_steps += 1

        if len(replay_buffer) >= batch_size and total_steps >= update_after:
            for _ in range(update_every):
                last_info = agent.update(replay_buffer, batch_size)

        if total_steps > 0 and total_steps % 5000 == 0:
            print(f"Saving checkpoint at step {total_steps}")
            agent.save(checkpoint_path)

        if done:
            print(
                f"Episode {episode_idx:04d} | "
                f"Return = {episode_return:10.3f} | "
                f"Steps = {episode_steps:4d} | "
                f"EEErr = {ee_err:8.3f} | "
                f"Buffer = {len(replay_buffer):6d}"
            )

            # reset onto a point on the reference circle using IK
            t = np.random.uniform(0.0, 2.0 * np.pi / omega)

            xd_init = np.array([
                cx + r * np.cos(omega * t),
                cy + r * np.sin(omega * t),
            ], dtype=np.float32)

            xd_init_next = np.array([
                cx + r * np.cos(omega * (t + dt)),
                cy + r * np.sin(omega * (t + dt)),
            ], dtype=np.float32)

            x_local_init = (xd_init[0] - arm.base[0]) / arm.scale
            y_local_init = (xd_init[1] - arm.base[1]) / arm.scale

            x_local_init_next = (xd_init_next[0] - arm.base[0]) / arm.scale
            y_local_init_next = (xd_init_next[1] - arm.base[1]) / arm.scale

            q_init = arm.ik4traject(x_local_init, y_local_init)
            q_init_next = arm.ik4traject(x_local_init_next, y_local_init_next)

            dq_init = (q_init_next - q_init) / dt
            dq_init = np.clip(dq_init, -10.0, 10.0).astype(np.float64)

            arm.q = q_init.astype(np.float64)
            arm.dq = np.zeros(2, dtype=np.float64)

            episode_steps = 0
            episode_return = 0.0
            episode_idx += 1

            ee_traj.clear()
            ref_traj.clear()

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

        if np.all(np.isfinite(xd)):
            pygame.draw.circle(screen, (220, 30, 30), xd.astype(int), 5)

        if np.all(np.isfinite(ee_next)):
            pygame.draw.circle(screen, (50, 200, 50), ee_next.astype(int), 5)

        q_text = font.render(
            f"q1={-arm.q[0]:.2f}, q2={-arm.q[1]:.2f}",
            True,
            (0, 0, 0),
        )
        screen.blit(q_text, (20, 20))

        step_text = font.render(
            f"total_steps={total_steps}, episode={episode_idx}, ep_steps={episode_steps}",
            True,
            (0, 0, 0),
        )
        screen.blit(step_text, (20, 45))

        reward_text = font.render(
            f"reward={reward:.3f}, ee_err={ee_err:.2f}",
            True,
            (0, 0, 0),
        )
        screen.blit(reward_text, (20, 70))

        action_text = font.render(
            f"tau=[{tau_env[0]:.2f}, {tau_env[1]:.2f}]",
            True,
            (0, 0, 0),
        )
        screen.blit(action_text, (20, 95))

        # target_text = font.render(
        #     f"q_des=[{q_des[0]:.2f}, {q_des[1]:.2f}] q_target=[{q_target[0]:.2f}, {q_target[1]:.2f}]",
        #     True,
        #     (0, 0, 0),
        # )
        # screen.blit(target_text, (20, 120))

        if last_info:
            actor_text = font.render(
                f"actor_loss={last_info.get('actor_loss', 0.0):.4f}, alpha={last_info.get('alpha', 0.0):.4f}",
                True,
                (0, 0, 0),
            )
            screen.blit(actor_text, (20, 145))

            critic_text = font.render(
                f"Q1_mean={last_info.get('q1_mean', 0.0):.4f}, Q2_mean={last_info.get('q2_mean', 0.0):.4f}",
                True,
                (0, 0, 0),
            )
            screen.blit(critic_text, (20, 170))

        pygame.display.flip()
        t += dt        
        clock.tick(60)

    print("Saving final checkpoint before exit...")
    agent.save(checkpoint_path)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()