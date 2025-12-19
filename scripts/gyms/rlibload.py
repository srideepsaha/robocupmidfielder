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
        metrics_logger.log_value("success", int(success), reduce="mean")

        # Handle episodes that may span iterations
        chunks = [episode] + (prev_episode_chunks or [])
        ep_return = float(sum(ch.get_return() for ch in chunks))

        # Print one line per finished episode (worker-side)
        er_id = kwargs.get("env_runner_id", "NA")
        print(f"[EP DONE er={er_id} pid={os.getpid()}] return={ep_return:.3f} success={int(success)}")

        # Sufficient stats for per-iteration mean/std on driver
        metrics_logger.log_value("episode_count",        1,                    reduce="sum")
        # metrics_logger.log_value("episode_return_sum",   ep_return,            reduce="sum", window=1)
        metrics_logger.log_value("episode_return_sum",   ep_return,            reduce="sum")
        metrics_logger.log_value("episode_return_sumsq", ep_return*ep_return,  reduce="sum")
        metrics_logger.log_value(
            "episode_returns_raw",
            ep_return,
            reduce="sum",
            clear_on_reduce=True
        )

    def on_train_result(self, *, algorithm, result, **kwargs):
        er = result.get("env_runners", {})
        s  = float(er.get("episode_return_sum",   0.0) or 0.0)
        ss = float(er.get("episode_return_sumsq", 0.0) or 0.0)
        n  = int(er.get("num_episodes",           0)   or 0)

        cm = result.setdefault("custom_metrics", {})
        if n > 0:
            mean = s / n
            var  = max(0.0, (ss / n) - mean * mean)  # population variance (use (ss - n*mean*mean)/(n-1) for sample)
            cm["episode_return_mean"] = mean
            cm["episode_return_std"]  = float(np.sqrt(var))
        else:
            cm["episode_return_mean"] = None
            cm["episode_return_std"]  = None

        print(f"[Custom metrics] Iter {algorithm.iteration} | "
              f"episodes={n} mean={cm['episode_return_mean']} std={cm['episode_return_std']}")




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




if __name__ == "__main__":
    import ray
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import time
    import torch

    # Shared setup
    NUM_ENVS = 8
    BASE_AGENT_PORT = 3100
    BASE_MONITOR_PORT = 3200
    EXTRA_PORTS = 2
    TOTAL_PORTS = NUM_ENVS + EXTRA_PORTS
    FREE_PORTS = [(3100 + i, 3200 + i) for i in range(TOTAL_PORTS)]

    def env_creator(env_ctx):
        idx = getattr(env_ctx, "worker_index")
        port_table = env_ctx["port_table"]
        agent_port, monitor_port = port_table[idx]
        return MultiAgentSharedWorld({
            "ip": "127.0.0.1",
            "robot_type": 0,
            "server_p": agent_port,
            "monitor_p": monitor_port,
        }, server_mgr=None)

    # === Phase 1: Train ===
    # print("[Phase 1] Training")
    # servers = Server(BASE_AGENT_PORT, BASE_MONITOR_PORT, TOTAL_PORTS)
    # ray.init()
    # register_env("MultiAgentSharedWorld", env_creator)

    # obs_space = gym.spaces.Box(-10, 10, shape=(3,), dtype=np.float32)
    # act_space = gym.spaces.Discrete(3)

    # cfg = (
    #     PPOConfig()
    #     .environment(
    #         env="MultiAgentSharedWorld",
    #         env_config={
    #             "ip": "127.0.0.1",
    #             "robot_type": 0,
    #             "port_table": FREE_PORTS,
    #         },
    #     )
    #     .multi_agent(
    #         policies={"shared_policy": (None, obs_space, act_space, {})},
    #         policy_mapping_fn=lambda aid, _: "shared_policy",
    #     )
    #     .framework("torch")
    #     .training(train_batch_size_per_learner=4096,
    #               minibatch_size=512,
    #               num_epochs=10,
    #               lr=3e-4,
    #               entropy_coeff=0.05)
    #     .env_runners(num_env_runners=NUM_ENVS,
    #                  num_envs_per_env_runner=1)
    #     .resources(num_gpus=0)
    # )

    # algo = cfg.build()
    # checkpoint_path = None
    # for i in range(200):
    #     res = algo.train()
    #     mean_r = (
    #         res.get("env_runners", {})
    #         .get("module_episode_returns_mean", {})
    #         .get("shared_policy", None)
    #     )
    #     if mean_r is not None:
    #         print(f"Iter {i+1:03d} – shared_policy reward: {mean_r:.3f}")
    #     if i == 4:
    #         checkpoint_path = algo.save_to_path()
    #         print(f"[Checkpoint] Saved to {checkpoint_path}")
    #         break

    # algo.stop()
    # ray.shutdown()
    # servers.kill()
    # time.sleep(2)

    # === Phase 2: Evaluate ===

    # eval_cfg = (
    #     PPOConfig()
    #     .environment(
    #         env="MultiAgentSharedWorld",
    #         env_config={
    #             "ip": "127.0.0.1",
    #             "robot_type": 0,
    #             "port_table": [(3109, 3209)],  # 👈 only one port pair
    #         },
    #     )
    #     .multi_agent(
    #         policies={"shared_policy": (None, obs_space, act_space, {})},
    #         policy_mapping_fn=lambda aid, _: "shared_policy",
    #     )
    #     .framework("torch")
    #     .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
    #     .resources(num_gpus=0)
    # )

    print("\n[Phase 2] Evaluation")

    from pathlib import Path
    from pyarrow.fs import LocalFileSystem

    servers = Server(BASE_AGENT_PORT, BASE_MONITOR_PORT, TOTAL_PORTS)
    # servers = Server(3109, 3209, 1)
    CHECKPOINT_DIR = Path("checkpoints_len_min")
    ckpt_uri = CHECKPOINT_DIR.resolve().as_uri()

    ray.init()
    from ray.tune.registry import register_env
    register_env("MultiAgentSharedWorld", env_creator)
    from ray.rllib.algorithms.algorithm import Algorithm
    restored_algo = Algorithm.from_checkpoint(ckpt_uri)
    module = restored_algo.get_module("shared_policy")

    env = MultiAgentSharedWorld({
        "ip": "127.0.0.1",
        "robot_type": 0,
        "server_p": 3109,
        "monitor_p": 3209,
    }, server_mgr=None)

    obs, _ = env.reset()
    done = {aid: False for aid in env.agents}
    done["__all__"] = False

    robot0_positions, robot1_positions, ball_positions = [], [], []

    while not done["__all__"]:
        action_dict = {}
        for aid, ob in obs.items():
            obs_tensor = torch.tensor([ob], dtype=torch.float32)
            logits = module.forward_inference({"obs": obs_tensor})["action_dist_inputs"]
            action = torch.argmax(logits, dim=-1).item()
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

    # Plot
    import pandas as pd
    robot0_positions = np.array(robot0_positions)
    robot1_positions = np.array(robot1_positions)
    ball_positions = np.array(ball_positions)

    # df = pd.DataFrame({
    #     "robot0_x": robot0_positions[:, 0],
    #     "robot0_y": robot0_positions[:, 1],
    #     "robot1_x": robot1_positions[:, 0],
    #     "robot1_y": robot1_positions[:, 1],
    #     "ball_x": ball_positions[:, 0],
    #     "ball_y": ball_positions[:, 1],
    # })

    # # Save to CSV
    # df.to_csv("trajectory_positions.csv", index=False)

    # Skip first 20 points
    robot0_positions = robot0_positions[20:]
    robot1_positions = robot1_positions[20:]
    ball_positions = ball_positions[20:]

    # plt.figure(figsize=(8, 6))
    # plt.plot(robot0_positions[:, 0], robot0_positions[:, 1], label='Robot 0')
    # plt.plot(robot1_positions[:, 0], robot1_positions[:, 1], label='Robot 1')
    # plt.plot(ball_positions[:, 0], ball_positions[:, 1], label='Ball', linestyle='--', color='orange')
    # plt.scatter(robot0_positions[0, 0], robot0_positions[0, 1], marker='o', color='blue', label='Robot 0 Start')
    # plt.scatter(robot1_positions[0, 0], robot1_positions[0, 1], marker='o', color='green', label='Robot 1 Start')
    # plt.scatter(ball_positions[0, 0], ball_positions[0, 1], marker='o', color='orange', label='Ball Start')
    # plt.legend()
    # plt.title("Trajectories from Restored Checkpoint")
    # plt.xlabel("X Position")
    # plt.ylabel("Y Position")
    # plt.grid(True)
    # plt.savefig("new_rollout_from_checkpoint.png")

    plt.figure(figsize=(8, 6))

    # Trajectories
    plt.plot(robot0_positions[:, 0], robot0_positions[:, 1], label='Robot 0', color='blue')
    plt.plot(robot1_positions[:, 0], robot1_positions[:, 1], label='Robot 1', color='green')
    plt.plot(ball_positions[:, 0], ball_positions[:, 1], label='Ball', linestyle='--', color='orange')

    # Start points with black edge
    plt.scatter(robot0_positions[0, 0], robot0_positions[0, 1],
                marker='o', color='blue', edgecolor='black', s=100, label='Robot 0 Start')
    plt.scatter(robot1_positions[0, 0], robot1_positions[0, 1],
                marker='o', color='green', edgecolor='black', s=100, label='Robot 1 Start')
    plt.scatter(ball_positions[0, 0], ball_positions[0, 1],
                marker='o', color='orange', edgecolor='black', s=100, label='Ball Start')

    # Labels and style
    plt.legend()
    plt.title("Trajectories from Restored Checkpoint")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)

    plt.savefig("new_rollout_from_checkpoint.png")


    restored_algo.stop()
    servers.kill()
    ray.shutdown()




