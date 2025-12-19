import gymnasium as gym
import numpy as np
import random
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from agent.Base_Agent import Base_Agent
from world.commons.Draw import Draw


class MultiAgentSharedWorld(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        ip, server_p, monitor_p, robot_type = (
            config["ip"],
            config["server_p"],
            config["monitor_p"],
            config["robot_type"],
        )

        self.agents = {
            "agent_0": Base_Agent(ip, server_p, monitor_p, 1, robot_type, "Team", True, False),
            "agent_1": Base_Agent(ip, server_p, monitor_p, 2, robot_type, "Team", True, False),
        }

        # RL parameters
        self.step_count = 0
        self.max_steps  = 500

        # Gym spaces
        self.observation_space = {
            aid: gym.spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
            for aid in self.agents
        }
        self.action_space = {aid: gym.spaces.Discrete(2) for aid in self.agents}

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
        positions = [(-3.0, -0.5, 0.5), (-3.0, 0.5, 0.5)]
        settle_y  = [-0.5, 0.5]

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
        ball_pos = (0.0, random.uniform(-0.5, 0.5), 0.0)
        self.agents["agent_0"].scom.unofficial_move_ball(ball_pos, (-4.0, 0.0, 0.0))
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
        return rel_ball.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Gymnasium step()
    # ------------------------------------------------------------------ #
    def step(self, action_dict):
        self.step_count += 1

        # ----- issue commands -----
        for aid, action in action_dict.items():
            a = self.agents[aid]
            if action == 1:
                a.behavior.execute("Walk", np.array([0.0, -0.5]), False, 0.0, True, 0.5)
                # a.behavior.execute("Zero")
                a.scom.commit_and_send(a.world.robot.get_command())
            else:
                a.behavior.execute("Walk", np.array([0.0, 0.5]), False, 0.0, True, 0.5)
                # a.behavior.execute("Zero")
                a.scom.commit_and_send(a.world.robot.get_command())
            
        self._recv_all()

        # ----- compute obs / rewards -----
        ball = self.agents["agent_0"].world.ball_abs_pos[:2]
        obs, rewards, infos = {}, {}, {}
        for aid, a in self.agents.items():
            dist = np.linalg.norm(a.world.robot.loc_head_position[:2] - ball)
            obs[aid]      = self._observe(a)
            rewards[aid]  = -dist
            infos[aid]    = {"distance_to_ball": dist}

        # bonus for closest
        closest = min(rewards, key=rewards.get)
        if rewards[closest] > -0.8:
            rewards[closest] += 5.0

        # ----- termination / truncation flags -----
        terminations = {aid: False for aid in self.agents}
        truncations  = {aid: False for aid in self.agents}
        # print("Here")

        if rewards[closest] > -0.8:                 # reached ball
            terminations[closest] = True
        if self.step_count >= self.max_steps:       # time limit
            truncations = {aid: True for aid in self.agents}

        terminations["__all__"] = any(terminations.values())
        truncations["__all__"]  = any(truncations.values())
        
        if terminations["__all__"] or truncations["__all__"]:
            print(f"[Episode finished] step={self.step_count}  "
                f"closest={closest}  reward={rewards[closest]:.2f}")

        for aid in infos:
            infos[aid]["episode"] = {"r": rewards[aid]}

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    def close(self):
        Draw.clear_all()
        for a in self.agents.values():
            a.terminate()


# ---------------------------------------------------------------------- #
# RLlib registration & run loop
# ---------------------------------------------------------------------- #
def env_creator(cfg):
    wid = cfg.get("worker_index", 0)
    offset = wid * 10
    return MultiAgentSharedWorld(
        {
            "ip": cfg["ip"],
            "robot_type": cfg["robot_type"],
            "server_p": cfg["server_p"] + offset,
            "monitor_p": cfg["monitor_p"] + offset,
        }
    )

if __name__ == "__main__":
    ray.init()

    obs_space = gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    register_env("MultiAgentSharedWorld", env_creator)

    cfg = (
        PPOConfig()
        .environment(
            env="MultiAgentSharedWorld",
            env_config=dict(ip="127.0.0.1", robot_type=0, server_p=3100, monitor_p=3200),
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda aid, episode: "shared_policy",
        )
        .framework("torch")
        .training(train_batch_size_per_learner=1, minibatch_size=1, num_epochs=10, lr=3e-4)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1, batch_mode="complete_episodes")
        .reporting(metrics_num_episodes_for_smoothing=1)
        .resources(num_gpus=0)
    )

    algo = cfg.build()
    try:
        for i in range(200):
            res = algo.train()
            reward = res.get("episode_reward_mean", None)
            # print(res["episodes_this_iter"])

            # Try to extract the shared_policy's reward from env_runners dict
            mean_r = (
                res.get("env_runners", {})
                .get("module_episode_returns_mean", {})
                .get("shared_policy", None)
            )

            if mean_r is not None:
                print(f"Iter {i+1:03d} – shared_policy reward: {mean_r:.3f}")
            else:
                pass

            # if reward is not None:
            #     print(f"Iter {i+1:03d} – mean reward: {reward:.3f}")
            # else:
            #     print(f"Iter {i+1:03d} – reward not available. Keys: {list(res.keys())}")

            #print(f"Iter: {i+1}, reward_mean: {result['episode_reward_mean']}")
            #print(f"Iter {i+1:03d} – mean reward: {res['episode_reward_mean']:.3f}")
    finally:
        algo.stop()
