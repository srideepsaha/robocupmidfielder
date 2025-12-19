import gymnasium as gym
import numpy as np
import random
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from agent.Base_Agent import Base_Agent
from world.commons.Draw import Draw
from scripts.commons.Server import Server
import torch

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode


import os
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import os
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class ForceCustomMetricCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, metrics_logger, prev_episode_chunks=None, **kwargs):
        # Success metric (mean over episodes)
        infos = episode.get_infos()
        last_infos = {aid: arr[-1] for aid, arr in infos.items() if arr}
        success = any(info.get("success", 0) for info in last_infos.values())
        metrics_logger.log_value("success", int(success), reduce="mean", clear_on_reduce=True)

        # Handle episodes that may span iterations
        chunks = [episode] + (prev_episode_chunks or [])
        ep_return = float(sum(ch.get_return() for ch in chunks))

        # Print one line per finished episode (worker-side)
        # er_id = kwargs.get("env_runner_id", "NA")
        # print(f"[EP DONE er={er_id} pid={os.getpid()}] return={ep_return:.3f} success={int(success)}")

        # Sufficient stats for per-iteration mean/std on driver
        metrics_logger.log_value("episode_count",        1,                    reduce="sum", clear_on_reduce=True)
        # metrics_logger.log_value("episode_return_sum",   ep_return,            reduce="sum", window=1)
        metrics_logger.log_value("episode_return_sum",   ep_return,            reduce="sum", clear_on_reduce=True)
        metrics_logger.log_value("episode_return_sumsq", ep_return*ep_return,  reduce="sum", clear_on_reduce=True)
        metrics_logger.log_value(
            "episode_returns_raw",
            ep_return,
            reduce="sum",
            clear_on_reduce=True
        )
        ep_len = sum(ch.env_steps() for ch in [episode] + (prev_episode_chunks or []))
        # ep_len = sum(ch.env_steps() for ch in [episode])
        metrics_logger.log_value("episode_len_sum", ep_len, reduce="sum", clear_on_reduce=True)
        metrics_logger.log_value("episode_len_sumsq", ep_len * ep_len, reduce="sum", clear_on_reduce=True)

    # def on_train_result(self, *, algorithm, result, **kwargs):
    #     er = result.get("env_runners", {})
    #     s  = float(er.get("episode_return_sum",   0.0) or 0.0)
    #     ss = float(er.get("episode_return_sumsq", 0.0) or 0.0)
    #     n  = int(er.get("num_episodes",           0)   or 0)

    #     cm = result.setdefault("custom_metrics", {})
    #     if n > 0:
    #         mean = s / n
    #         var  = max(0.0, (ss / n) - mean * mean)  # population variance (use (ss - n*mean*mean)/(n-1) for sample)
    #         cm["episode_return_mean"] = mean
    #         cm["episode_return_std"]  = float(np.sqrt(var))
    #     else:
    #         cm["episode_return_mean"] = None
    #         cm["episode_return_std"]  = None

    #     print(f"[Custom metrics] Iter {algorithm.iteration} | "
    #           f"episodes={n} mean={cm['episode_return_mean']} std={cm['episode_return_std']}")

    def on_train_result(self, *, algorithm, result, **kwargs):
        er = result.get("env_runners", {})

        # ---- returns ----
        s_ret  = float(er.get("episode_return_sum",   0.0) or 0.0)
        ss_ret = float(er.get("episode_return_sumsq", 0.0) or 0.0)

        # ---- lengths (make sure you logged these in on_episode_end) ----
        s_len  = float(er.get("episode_len_sum",      0.0) or 0.0)
        ss_len = float(er.get("episode_len_sumsq",    0.0) or 0.0)

        n = int(er.get("num_episodes", 0) or 0)

        cm = result.setdefault("custom_metrics", {})
        if n > 0:
            # returns
            mean_r = s_ret / n
            var_r  = max(0.0, (ss_ret / n) - mean_r * mean_r)  # population variance
            cm["episode_return_mean"] = mean_r
            cm["episode_return_std"]  = float(np.sqrt(var_r))

            # lengths
            mean_l = s_len / n
            var_l  = max(0.0, (ss_len / n) - mean_l * mean_l)  # population variance
            cm["episode_len_mean_custom"] = mean_l
            cm["episode_len_std"]         = float(np.sqrt(var_l))
        else:
            cm["episode_return_mean"] = None
            cm["episode_return_std"]  = None
            cm["episode_len_mean_custom"] = None
            cm["episode_len_std"]         = None

        print(
            f"[Custom metrics] Iter {algorithm.iteration} | "
            f"episodes={n} "
            f"ret(mean={cm['episode_return_mean']}, std={cm['episode_return_std']}) "
            f"len(mean={cm['episode_len_mean_custom']}, std={cm['episode_len_std']})"
        )





# class ForceCustomMetricCallback(DefaultCallbacks):
#     def on_episode_end(self, *, episode, metrics_logger, prev_episode_chunks=None, **kwargs):
#         # Success metric (mean over episodes)
#         infos = episode.get_infos()
#         last_infos = {aid: arr[-1] for aid, arr in infos.items() if arr}
#         success = any(info.get("success", 0) for info in last_infos.values())
#         metrics_logger.log_value("success", int(success), reduce="mean")

#         # Total episode return (handles chunked episodes)
#         chunks = [episode] + (prev_episode_chunks or [])
#         ep_return = float(sum(ch.get_return() for ch in chunks))

#         # Log sufficient statistics (safe to aggregate across env-runners)
#         metrics_logger.log_value("episode_count", 1, reduce="sum", window=1)
#         metrics_logger.log_value("episode_return_sum", ep_return, reduce="sum", window=1)
#         metrics_logger.log_value("episode_return_sumsq", ep_return * ep_return, reduce="sum", window=1)


#     def on_train_result(self, *, algorithm, result, **kwargs):
#         er = result.get("env_runners", {})
#         s  = float(er.get("episode_return_sum", 0.0) or 0.0)
#         ss = float(er.get("episode_return_sumsq", 0.0) or 0.0)
#         n  = int(er.get("episode_count", 0) or 0)

#         cm = result.setdefault("custom_metrics", {})
#         if n > 0:
#             mean = s / n
#             var  = max(0.0, (ss / n) - mean * mean)  # population variance
#             cm["episode_return_mean"] = mean
#             cm["episode_return_std"]  = float(np.sqrt(var))
#         else:
#             cm["episode_return_mean"] = None
#             cm["episode_return_std"]  = None

#         print(f"[Custom metrics] Iteration: {algorithm.iteration}")
#         print(" STD episode_return_mean:", cm["episode_return_mean"])
#         print(" STD episode_return_std:",  cm["episode_return_std"])



# class ForceCustomMetricCallback(DefaultCallbacks):
#     def on_episode_end(self, *, episode: MultiAgentEpisode, metrics_logger, **kwargs):
#         infos_dict = episode.get_infos()
#         # print(f"[DEBUG][Callback] infos_dict: {infos_dict}") 

#         # Get last info entries for each agent
#         last_infos = {aid: infos[-1] for aid, infos in infos_dict.items() if infos}
#         # print(f"[DEBUG][Callback] last_infos: {last_infos}")

#         # Compute success based on those
#         success = any(info.get("success", 0) for info in last_infos.values())

#         metrics_logger.log_value("success", int(success), reduce="mean")

#     def on_train_result(self, *, algorithm, result, **kwargs):
#         print("Success stats this iter:", result.get("custom_metrics", {}))


class MultiAgentSharedWorld(MultiAgentEnv):
    def __init__(self, config, server_mgr=None):
        super().__init__()
        ip, server_p, monitor_p, robot_type = (
            config["ip"],
            config["server_p"],
            config["monitor_p"],
            config["robot_type"],
        )
        #idx = getattr(config, "worker_index", 0)
        #print(f"Init {idx}")
        self.agents = {
            "agent_0": Base_Agent(ip, server_p, monitor_p, 1, robot_type, "Team", True, False),
            "agent_1": Base_Agent(ip, server_p, monitor_p, 2, robot_type, "Team", True, False),
        }

        # RL parameters
        self.step_count = 0
        self.max_steps  = 1500

        # Gym spaces
        self.observation_space = {
            aid: gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
            for aid in self.agents
        }
        self.action_space = {aid: gym.spaces.Discrete(3) for aid in self.agents}

        self.server_mgr = server_mgr

        # for a in self.agents.values():
        #     a.scom.unofficial_set_play_mode("PlayOn")

    # ------------------------------------------------------------------ #
    # Helper: batch send / recv
    # ------------------------------------------------------------------ #
    def _send_all(self):
        for a in self.agents.values():
            a.scom.commit_and_send(a.world.robot.get_command())

    def _recv_all(self):
        for a in self.agents.values():
            a.scom.receive()

    # ------------------------------------------------------------------ #
    # Helper: teleport robots + ball and settle simulation
    # ------------------------------------------------------------------ #
    def teleport_robots_and_ball(self):
        positions = [(-3.0, -2.0, 0.5), (-3.0, 2.0, 0.5)]
        settle_y  = [-2.0, 2.0]

        # beam in air, settle, beam to ground ­– gives stable start pose
        for _ in range(25):
            for i, a in enumerate(self.agents.values()):
                a.scom.unofficial_beam(positions[i], 0)
                a.behavior.execute("Zero")
                a.scom.commit_and_send(a.world.robot.get_command())
            # self._send_all(); self._recv_all()
            self._recv_all()

        for i, a in enumerate(self.agents.values()):
            r = a.world.robot
            a.scom.unofficial_beam((-3.0, settle_y[i], a.world.robot.beam_height), 0)
            r.joints_target_speed[0] = 0.01
            a.behavior.execute("Zero")
            a.scom.commit_and_send(a.world.robot.get_command())
        self._recv_all()

        for _ in range(7):
            for a in self.agents.values():
                a.behavior.execute("Zero")
                a.scom.commit_and_send(a.world.robot.get_command())
            self._recv_all()

        # move ball
        x_pos = 0.0
        # y_pos = random.uniform(-1.5, 1.5)
        y_right = random.uniform(-3.5, -0.5)
        y_left = random.uniform(0.5, 3.5)
        # y_pos = random.choice([y_left, y_right])
        if random.choice([True, False]):
            y_pos = y_left
            # Use specific speed for left side
            speed_y = random.uniform(3.5 - y_pos, 0.5 - y_pos)  
        else:
            y_pos = y_right
            # Use specific speed for right side
            speed_y = random.uniform(-0.5 - y_pos, -3.5 - y_pos)  

        #speed_y = random.uniform(1.5 - y_pos, -1.5 - y_pos)
        ball_pos = (0.0, y_pos, 0.0)
        self.agents["agent_0"].scom.unofficial_move_ball(ball_pos, (-4.0, speed_y, 0.0))
        # self._send_all(); self._recv_all()

    # ------------------------------------------------------------------ #
    # per-agent space accessors (new Gymnasium/RLlib expectation)
    # ------------------------------------------------------------------ #
    def get_action_space(self, agent_id: str | None = None):
        return self.action_space if agent_id is None else self.action_space[agent_id]

    def get_observation_space(self, agent_id: str | None = None):
        return (
            self.observation_space
            if agent_id is None
            else self.observation_space[agent_id]
        )

    # ------------------------------------------------------------------ #
    # Gymnasium reset()
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.teleport_robots_and_ball()

        obs   = {aid: self._observe(a) for aid, a in self.agents.items()}
        infos = {aid: {}              for aid in self.agents}
        return obs, infos

    # ------------------------------------------------------------------ #
    # Build a 2-d relative-ball observation
    # ------------------------------------------------------------------ #
    def _observe(self, agent):
        w = agent.world
        ball_pos   = w.ball_abs_pos[:2]
        robot_pos  = w.robot.loc_head_position[:2]
        rel_ball   = ball_pos - robot_pos
        rel_ball_pos = w.ball_rel_torso_cart_pos[:2]
        init = [0.0,0.0]
        distance = np.linalg.norm(rel_ball_pos - init)
        ball_heading = np.arctan2(rel_ball_pos[1], rel_ball_pos[0])
        return np.array([rel_ball_pos[0], rel_ball_pos[1], ball_heading], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Gymnasium step()
    # ------------------------------------------------------------------ #
    def step(self, action_dict):
        self.step_count += 1

        # ----- issue commands -----
        for aid, action in action_dict.items():
            a = self.agents[aid]
            if action == 0:
                a.behavior.execute("Walk", np.array([0.0, -0.5]), False, 0.0, True, 0.5)
                # a.behavior.execute("Zero")
                a.scom.commit_and_send(a.world.robot.get_command())
            elif action == 1:
                a.behavior.execute("Walk", np.array([0.0, 0.5]), False, 0.0, True, 0.5)
                # a.behavior.execute("Zero")
                a.scom.commit_and_send(a.world.robot.get_command())
            else:
                a.behavior.execute("Walk", np.array([0.0, 0.0]), False, 0.0, True, 0.5)
                # a.behavior.execute("Zero")
                a.scom.commit_and_send(a.world.robot.get_command())
            
        self._recv_all()

        # ───── 1. build obs & core shaped reward ────────────────────────────
        ball = self.agents["agent_0"].world.ball_abs_pos[:2]

        obs, rewards, infos, distances = {}, {}, {}, {}
        for aid, a in self.agents.items():

            # observation (unchanged)
            obs[aid] = self._observe(a)
            # rel_ball = a.world.ball_rel_torso_cart_pos[:2]
            distance = np.linalg.norm(obs[aid][:2])
            heading = obs[aid][2]

            # save for later: distance ranking
            distances[aid] = distance

            # # -------- reward shaping like InterceptionEnv ----------
            # distance_reward = -distance                      # smaller is better
            # heading_reward  = -abs(heading)                  # closer to 0 rad is better
            # rewards[aid]    = 0.25 * distance_reward + 0.5 * heading_reward
            # # optional small time penalty to encourage speed
            # # rewards[aid]   += -0.01
            # distances[aid] = distance          

            infos[aid] = {"distance": distance, "heading": heading}

        # ───── 2. closest / farthest logic ──────────────────────────────────
        # closest_agent  = min(distances, key=distances.get)
        # farthest_agent = max(distances, key=distances.get)

        for aid in self.agents:
            if aid not in rewards:
                rewards[aid] = 0.0

        closest_agent  = min(obs, key=lambda aid: abs(obs[aid][2]))
        farthest_agent = max(obs, key=lambda aid: abs(obs[aid][2]))

        # print(f"Closest agent: {self.agents[closest_agent].unum}, Farthest agent: {self.agents[farthest_agent].unum}")


        for aid in self.agents:
            if aid == closest_agent:
                distance_reward = -distances[aid]
                heading_reward  = -abs(obs[aid][2])  # heading
                rewards[aid] = 0.3 * distance_reward + 0.7 * heading_reward
            # else:
            #     rewards[aid] = 0.0  # farthest agent gets no shaping reward
            #     if action != 2:
            #         rewards[aid] -= 0.35         # 0.4
            #     else:
            #         pass


        # Bonus for the closest
        # rewards[closest_agent] += 0.6

        # --- Penalize the farther agent for moving ---
        for aid, action in action_dict.items():
            if aid == farthest_agent:
                rewards[aid] = 0.0
                if action != 2:  # Not 'stand still'
                    rewards[aid] -= 0.5  # Penalty for moving when farther # -0.3
                # else:
                #     rewards[aid] += 0.1  # (Optional) tiny bonus for standing


        # Collision penalty
        robot_0_pos = self.agents["agent_0"].world.robot.loc_head_position[:2]
        robot_1_pos = self.agents["agent_1"].world.robot.loc_head_position[:2]
        if np.linalg.norm(robot_0_pos - robot_1_pos) < 0.3:
            for aid in self.agents:
                rewards[aid] -= 0.3

        # for aid, action in action_dict.items():
        #     if aid == farthest_agent:
        #         if action != 2:          # 2 = stand still
        #             rewards[aid] -= 0.4  # discourage wandering
        #         else:
        #             rewards[aid] += 0.1  # tiny bonus for staying put

        # ───── 3. success & failure bonuses  ────────────────────────────────
        terminations = {aid: False for aid in self.agents}
        truncations  = {aid: False for aid in self.agents}

        for aid in self.agents:
            dist = distances[aid]
            heading = obs[aid][2]

            # success condition (same numbers as single-agent env)
            if aid == closest_agent and dist < 0.8 and abs(heading) < 0.03:
                terminations[aid] = True
                rewards[aid] += 0.9
        
        # if terminations[aid] == True:
        #     rewards[farthest_agent] += 0.9

        if terminations[closest_agent]:
            rewards[farthest_agent] += 0.9

        # time-limit failure penalty
        if self.step_count >= self.max_steps:
            truncations = {aid: True for aid in self.agents}
            for aid in self.agents:
                if aid == closest_agent and abs(obs[aid][2]) > 0.025:     # didn’t align in time
                    rewards[aid] -= 0.4 #0.2

        # combine multi-agent flags
        terminations["__all__"] = any(terminations.values())
        truncations ["__all__"] = any(truncations .values())
        
        if terminations["__all__"] or truncations["__all__"]:
            # print(f"[Episode finished] step={self.step_count}")
            success = int(terminations["__all__"])
            for aid in self.agents:
                infos[aid]["success"] = success  # RLlib will pick this up
            # print(f"[DEBUG] Infos before return: {infos}")


            # print(f"[LOGGING SUCCESS = {int(terminations['__all__'])}]")

        # for aid in infos:
        #     infos[aid]["episode"] = {"r": rewards[aid]}
        #     # infos[aid]["episode"]["success"] = 1 if terminations[aid] else 0
        # print(f"[DEBUG] Final step info: {infos}")

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    def close(self):
        Draw.clear_all()
        for a in self.agents.values():
            a.terminate()
        if self.server_mgr is not None:
            self.server_mgr.kill()  # ← clean up server process


# ---------------------------------------------------------------------- #
# RLlib registration & run loop
# ---------------------------------------------------------------------- #
# Global counter to keep track of environment launches
# def env_creator(env_ctx):
#     base_port = 3100
#     monitor_base = 3200

#     wid = env_ctx.get("worker_idx", env_ctx.get("worker_index", 0))
#     idx = wid  # Or use env_runner_idx if needed

#     server_p  = base_port + idx
#     monitor_p = monitor_base + idx

#     print(f"[env_creator] Reusing server: ports {server_p}, {monitor_p}")

#     return MultiAgentSharedWorld(
#         {
#             "ip": "127.0.0.1",
#             "robot_type": 0,
#             "server_p": server_p,
#             "monitor_p": monitor_p,
#         },
#         server_mgr=None  # No auto-launch or cleanup
#     )




if __name__ == "__main__":
    import ray
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig
    import gymnasium as gym
    import time

    # ---------- 1. choose how many parallel envs you want ---------------
    NUM_ENVS          = 8                       # one SimSpark per env-runner
    BASE_AGENT_PORT   = 3100                    # SimSpark --agent-port
    BASE_MONITOR_PORT = 3200                    # SimSpark --server-port
    PORT_STRIDE       = 1                       # Server() already does +1
    EXTRA_PORTS = 2  # <--- Number of "spare" ports to avoid index errors
    FREE_PORTS = [(3100+i, 3200+i) for i in range(NUM_ENVS + EXTRA_PORTS)]
    TOTAL_PORTS = NUM_ENVS + EXTRA_PORTS


    # ---------- 2. launch ALL SimSpark servers up-front -----------------
    print(f"[main] Spawning {NUM_ENVS} SimSpark servers …")
    servers = Server(BASE_AGENT_PORT, BASE_MONITOR_PORT, TOTAL_PORTS)
    # Give SimSpark a split-second to bind the ports cleanly

    # ---------- 3. env_creator that just CONNECTS to the right port -----
    # def env_creator(env_ctx):
    #     print(f"[env_creator] env_ctx = {env_ctx}")

    #     base_agent_port = 3100
    #     base_monitor_port = 3200
    #     stride = 10

    #     wid = env_ctx.get("worker_idx", env_ctx.get("worker_index", 0))
    #     vid = env_ctx.get("env_runner_idx", env_ctx.get("vector_index", 0))

    #     idx = wid + vid
    #     agent_port = base_agent_port + idx * stride
    #     monitor_port = base_monitor_port + idx * stride

    #     print(f"[env_creator] idx={idx}, agent_port={agent_port}, monitor_port={monitor_port}")

    #     # server_mgr = Server(agent_port, monitor_port, 1)
    #     print(f"[env_creator] Server launched on ports {agent_port}, {monitor_port}")

    #     return MultiAgentSharedWorld(
    #         {
    #             "ip": "127.0.0.1",
    #             "robot_type": 0,
    #             "server_p": agent_port,
    #             "monitor_p": monitor_port,
    #         },
    #         server_mgr=server_mgr
    #     )

    # def env_creator(env_ctx):
    #     import os

    #     manual_ports = [
    #         (3100, 3200),
    #         (3101, 3201),
    #         (3102, 3202),
    #         (3103, 3203),
    #     ]

    #     wid = env_ctx.get("worker_index", 0)
    #     idx = wid  # one env per runner


    #     if idx >= len(manual_ports):
    #         raise RuntimeError(f"Too many envs requested! idx={idx} but only {len(manual_ports)} servers started.")

    #     agent_port, monitor_port = manual_ports[idx]
    #     print(f"[env_creator] idx={idx}, agent_port={agent_port}, monitor_port={monitor_port}")
    #     print(f"[DEBUG] PID={os.getpid()} – Starting env idx={idx} (agent_port={agent_port})")
    #     print(f"[env_creator] idx={idx}, ports={manual_ports[idx]}, PID={os.getpid()}")


    #     return MultiAgentSharedWorld(
    #         {
    #             "ip": "127.0.0.1",
    #             "robot_type": 0,
    #             "server_p": agent_port,
    #             "monitor_p": monitor_port,
    #         },
    #         server_mgr=None,  # Already launched externally
    #     )


    # def env_creator(env_ctx):
    #     print("ENV_CTX:", env_ctx)
    #     runner_id = env_ctx.get("env_runner_idx") or env_ctx.get("vector_index") or env_ctx.get("worker_index") or 0
    #     port_table = env_ctx["port_table"]

    #     agent_p, monitor_p = port_table[runner_id]
    #     print(f"[env_creator] runner {runner_id} → {agent_p}/{monitor_p}")

    #     return MultiAgentSharedWorld(
    #         dict(ip="127.0.0.1", robot_type=0,
    #             server_p=agent_p, monitor_p=monitor_p),
    #         server_mgr=None,
    #     )

    def env_creator(env_ctx):
        # env_ctx["worker_index"] is 0-based
        idx = getattr(env_ctx, "worker_index")
        port_table = env_ctx["port_table"]
        agent_port, monitor_port = port_table[idx]
        print(f"The index is {idx}, port_table is {port_table}")
        print(f"env_creator: idx={idx}  port_table_len={len(port_table)}  port_table={port_table}")
        print(f"[ENV_CREATOR DEBUG] idx={idx} of {len(port_table)}, agent_port={agent_port}, monitor_port={monitor_port}")

        # Now pass the correct port info to your custom MultiAgentEnv
        return MultiAgentSharedWorld(
            {
                "ip": "127.0.0.1",
                "robot_type": 0,
                "server_p": agent_port,
                "monitor_p": monitor_port,
            },
            server_mgr=None,
        )


    # ---------- 4. RLlib setup ------------------------------------------
    ray.init(log_to_driver=True)

    register_env("MultiAgentSharedWorld", env_creator)

    obs_space = gym.spaces.Box(-10, 10, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    cfg = (
        PPOConfig()
        .environment(
            env="MultiAgentSharedWorld",
            env_config={
                "ip": "127.0.0.1",
                "robot_type": 0,
                "port_table": FREE_PORTS,
            },
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda aid, _: "shared_policy",
        )
        .framework("torch")
        .training(train_batch_size_per_learner=4096,   #4096
                  minibatch_size=512,
                  num_epochs=10,
                  lr=3e-4,
                  entropy_coeff=0.05)
        .env_runners(batch_mode="complete_episodes",num_env_runners=NUM_ENVS,
                     num_envs_per_env_runner=1)
        .resources(num_gpus=0)
        .callbacks(ForceCustomMetricCallback)
        #
    )


    # ---------- 5. training loop ----------------------------------------

    from pathlib import Path
    from pyarrow.fs import LocalFileSystem

    import pandas as pd
    import csv, os

    CSV_PATH = "training_metrics.csv"
    new_file = not os.path.exists(CSV_PATH)
    csv_f = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_f)

    if new_file:
        writer.writerow([
            "iteration",
            "episode_len_mean",
            "episode_return_mean",
            "episode_return_sum",
            "success_mean",
            "num_episodes",
            "num_env_steps_sampled",
            "num_env_steps_sampled_lifetime",
            "cm_episode_return_mean",
            "cm_episode_return_std",
            "cm_episode_len_std",
            "cm_episode_len_mean_custom",
        ])

    def to_uri(p: Path) -> str:
        return p.resolve().as_uri()

    CHECKPOINT_DIR = Path("checkpoints_len_min")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ep_len = None
    best_ckpt_path = None
    EVAL_AGENT_PORT, EVAL_MONITOR_PORT = 3999, 4999
    eval_server = Server(EVAL_AGENT_PORT, EVAL_MONITOR_PORT, 1)
    algo = cfg.build()
    try:
        for i in range(100):
            res = algo.train()
            mean_r = (
                res.get("env_runners", {})
                   .get("module_episode_returns_mean", {})
                   .get("shared_policy", None)
            )
            if mean_r is not None:
                print(f"Iter {i+1:03d} – shared_policy reward: {mean_r:.3f}")

            success = res.get("custom_metrics", {}).get("success_mean", None)
            if success is not None:
                print(f"Iter {i:03d} – success rate: {success:.2%}")
            # print("res keys:", res.keys())
            # print_nested(res)
            # print("Episodes completed:", res.get("episodes_total", "n/a"))
            # print("env_runners:", res.get("env_runners", {}))
            # print("env_runners:", res.get("learners", {}))
            envr = res.get("env_runners", {})
            cm   = res.get("custom_metrics", {}) or {}
            print(f"Episodes this iter: {envr.get('num_episodes')}")
            print(f"Lifetime episodes: {envr.get('num_episodes_lifetime')}")
            print(f"Shared policy mean reward: {envr.get('module_episode_returns_mean', {}).get('shared_policy')}")
            print(f"Overall mean ep return: {envr.get('episode_return_mean')}")

            print(f"Overall success rate: {envr.get('success', 'N/A')}")
            print(f"Num env steps lifetime: {envr.get('num_env_steps_sampled_lifetime', 'N/A')}")
            # Step-related metrics
            print(f"Mean episode length: {envr.get('episode_len_mean', 'N/A')}")
            # print(f"Max episode length: {envr.get('episode_len_max', 'N/A')}")
            # print(f"Min episode length: {envr.get('episode_len_min', 'N/A')}")
            ep_return_mean = envr.get("episode_return_mean")
            ep_len_mean = envr.get("episode_len_mean")
            num_eps = envr.get("num_episodes")
            num_env_steps = envr.get("num_env_steps_sampled")

            if ep_return_mean is not None and ep_len_mean not in (None, 0):
                mean_r_per_step = ep_return_mean / ep_len_mean
                steps_this_iter = num_env_steps if num_env_steps is not None else (
                    (ep_len_mean or 0) * (num_eps or 0)
                )
                total_reward_est = mean_r_per_step * (steps_this_iter or 0)
            else:
                total_reward_est = "N/A"

            print(f"Total reward this iter (est., incl. partial): {total_reward_est}")

            if ep_len_mean is not None and (best_ep_len is None or ep_len_mean < best_ep_len):
                best_ep_len = ep_len_mean

                fs = LocalFileSystem()
                ckpt_path = algo.save_to_path(str(CHECKPOINT_DIR.resolve()), filesystem=fs) 
                print(f"[Checkpoint] Saved to {ckpt_path}")

            writer.writerow([
                i + 1,
                envr.get("episode_len_mean") if envr.get("episode_len_mean") is not None else "",
                envr.get("episode_return_mean") if envr.get("episode_return_mean") is not None else "",
                envr.get("episode_return_sum") if envr.get("episode_return_sum") is not None else "",
                envr.get("success") if envr.get("success") is not None else "",
                envr.get("num_episodes") if envr.get("num_episodes") is not None else "",
                envr.get("num_env_steps_sampled") if envr.get("num_env_steps_sampled") is not None else "",
                envr.get("num_env_steps_sampled_lifetime") if envr.get("num_env_steps_sampled_lifetime") is not None else "",
                cm.get("episode_return_mean") if cm.get("episode_return_mean") is not None else "",
                cm.get("episode_return_std") if cm.get("episode_return_std") is not None else "",
                cm.get("episode_len_std") if cm.get("episode_len_std") is not None else "",
                cm.get("episode_len_mean_custom") if cm.get("episode_len_mean_custom") is not None else "",
            ])
            csv_f.flush()


            # Every 50 iterations, plot a rollout episode
            if (i+1) % 10 == 0:
                print(f"\n[Eval] Plotting rollout for iteration {i+1}")
                # ---------- Rollout ----------
                # env = MultiAgentSharedWorld({
                #     "ip": "127.0.0.1",
                #     "robot_type": 0,
                #     "server_p": 3109,   # Make sure ports are free!
                #     "monitor_p": 3209,
                # }, server_mgr=None)
                EVAL_AGENT_PORT, EVAL_MONITOR_PORT = 3999, 4999
                env = MultiAgentSharedWorld({
                    "ip": "127.0.0.1",
                    "robot_type": 0,
                    "server_p": EVAL_AGENT_PORT,
                    "monitor_p": EVAL_MONITOR_PORT,
                }, server_mgr=None)

                obs, info = env.reset()
                done = {aid: False for aid in env.agents}
                done["__all__"] = False

                robot0_positions, robot1_positions, ball_positions = [], [], []
                module = algo.get_module("shared_policy")
                while not done["__all__"]:
                    action_dict = {}
                    for aid, ob in obs.items():
                        ob = np.asarray(ob, dtype=np.float32)
                        obs_tensor = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
                        result = module.forward_inference({"obs": obs_tensor})
                        # print("forward_inference result:", result)
                        logits = result["action_dist_inputs"]  # This is a tensor of shape [1, num_actions]
                        action = torch.argmax(logits, dim=-1).item()  # Greedy action selection (for evaluation)

                        action_dict[aid] = action

                    obs, rewards, term, trunc, info = env.step(action_dict)
                    done = {**term, **trunc}
                    done["__all__"] = term.get("__all__", False) or trunc.get("__all__", False)

                    robot0 = env.agents["agent_0"].world.robot.loc_head_position[:2]
                    robot1 = env.agents["agent_1"].world.robot.loc_head_position[:2]
                    ball   = env.agents["agent_0"].world.ball_abs_pos[:2]
                    robot0_positions.append(robot0.copy())
                    robot1_positions.append(robot1.copy())
                    ball_positions.append(ball.copy())
                env.close()

                # ---------- Plot ----------
                import matplotlib.pyplot as plt
                robot0_positions = np.array(robot0_positions)
                robot1_positions = np.array(robot1_positions)
                ball_positions = np.array(ball_positions)

                plt.figure(figsize=(8,6))
                plt.plot(robot0_positions[:,0], robot0_positions[:,1], label='Robot 0')
                plt.plot(robot1_positions[:,0], robot1_positions[:,1], label='Robot 1')
                plt.plot(ball_positions[:,0], ball_positions[:,1], label='Ball', linestyle='--', color='orange')
                plt.scatter(robot0_positions[0,0], robot0_positions[0,1], marker='o', color='blue', label='Robot 0 Start')
                plt.scatter(robot1_positions[0,0], robot1_positions[0,1], marker='o', color='green', label='Robot 1 Start')
                plt.scatter(ball_positions[0,0], ball_positions[0,1], marker='o', color='orange', label='Ball Start')
                plt.legend()
                plt.title(f"Trajectories at Iteration {i+1}")
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
                plt.grid(True)
                plt.savefig(f"rollout_for_episode{i+1}.png")

    finally:
        print("\n[main] Shutting down …")
        algo.stop()
        servers.kill()      # <- terminate all SimSpark processes
        ray.shutdown()
        csv_f.close()



