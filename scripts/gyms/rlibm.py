import gymnasium as gym
import numpy as np
import random
import threading
import queue
import time
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from agent.Base_Agent import Base_Agent
from world.commons.Draw import Draw


class AgentThreadedComm:
    def __init__(self, agent):
        self.agent = agent
        self.command_queue = queue.Queue()
        self.latest_obs = None
        self.lock = threading.Lock()
        self.stop_flag = False
        self.thread = threading.Thread(target=self._comm_loop, daemon=True)
        self.thread.start()

    def _comm_loop(self):
        while not self.stop_flag:
            try:
                cmd = self.command_queue.get(timeout=0.05)
            except queue.Empty:
                cmd = None

            try:
                if cmd is not None:
                    self.agent.scom.commit(cmd)
                self.agent.scom.commit(b'(syn)')  # Always send (syn) for sync mode
                self.agent.scom.send()
                self.agent.scom.receive()

                with self.lock:
                    self.latest_obs = self._build_observation()

            except Exception as e:
                agent_id = getattr(self.agent, 'unum', 'unknown')
                print(f"Agent {agent_id} communication error: {e}")

    def _build_observation(self):
        r = self.agent.world.robot
        obs = np.zeros(70, dtype=np.float32)
        obs[0] = 0
        obs[1] = r.loc_head_z * 3
        obs[2] = r.loc_head_z_vel / 2
        obs[3] = r.imu_torso_orientation / 50
        obs[4] = r.imu_torso_roll / 15
        obs[5] = r.imu_torso_pitch / 15
        obs[6:9] = r.gyro / 100
        obs[9:12] = r.acc / 10
        obs[12:18] = r.frp.get('lf', (0, 0, 0, 0, 0, 0))
        obs[18:24] = r.frp.get('rf', (0, 0, 0, 0, 0, 0))
        obs[15:18] /= 100
        obs[21:24] /= 100
        obs[24:44] = r.joints_position[2:22] / 100
        obs[44:64] = r.joints_speed[2:22] / 6.1395
        obs[64:70] = 0
        return obs

    def send_command(self, cmd):
        self.command_queue.put(cmd)

    def get_observation(self):
        with self.lock:
            return self.latest_obs

    def stop(self):
        self.stop_flag = True
        self.thread.join()


class MultiAgentSharedWorld(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        ip = config["ip"]
        server_p = config["server_p"]
        monitor_p = config["monitor_p"]
        robot_type = config["robot_type"]

        self.agents = {
            "agent_0": Base_Agent(ip, server_p, monitor_p, 4, robot_type, "Gym", True, False),
            "agent_1": Base_Agent(ip, server_p, monitor_p, 5, robot_type, "Gym", True, False)
        }

        self.comm_threads = {
            aid: AgentThreadedComm(agent)
            for aid, agent in self.agents.items()
        }

        self.step_count = 0
        self.max_steps = 300

        self.observation_space = {
            aid: gym.spaces.Box(low=-10, high=10, shape=(70,), dtype=np.float32)
            for aid in self.agents.keys()
        }
        self.action_space = {
            aid: gym.spaces.Discrete(2)
            for aid in self.agents.keys()
        }

    def teleport_robots_and_ball(self):
        positions = [(-3.0, -1.5, 0.5), (-3.0, 1.5, 0.5)]
        for i, (aid, agent) in enumerate(self.agents.items()):
            # Send unofficial beam command to monitor socket (no receive here)
            agent.scom.unofficial_beam(positions[i], 0)
            # Send robot command through comm thread queue
            cmd = agent.world.robot.get_command()
            self.comm_threads[aid].send_command(cmd)

        time.sleep(0.5)  # Wait for comm threads to process beam commands

        # Move the ball - only monitor socket, no receive
        x_pos = 0.0
        y_pos = random.uniform(-0.5, 0.5)
        ball_pos = (x_pos, y_pos, 0.0)
        ball_vel = (0.0, 0.0, 0.0)
        self.agents["agent_0"].scom.unofficial_move_ball(ball_pos, ball_vel)

        time.sleep(0.2)  # Small sleep to ensure ball move is registered

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.teleport_robots_and_ball()

        for aid, agent in self.agents.items():
            agent.behavior.execute("Zero")
            cmd = agent.world.robot.get_command()
            self.comm_threads[aid].send_command(cmd)

        time.sleep(1)  # Allow threads to process commands and update observations

        obs = {}
        for aid, comm_thread in self.comm_threads.items():
            ob = comm_thread.get_observation()
            if ob is None:
                ob = np.zeros(70, dtype=np.float32)
            obs[aid] = ob
        return obs, {}  # Gymnasium API compliance

    def step(self, action_dict):
        self.step_count += 1

        for aid, action in action_dict.items():
            agent = self.agents[aid]
            if action == 0:
                agent.behavior.execute("Zero")
            elif action == 1:
                agent.behavior.execute("Walk", np.array([0.2, 0.0]), False, 0.0, True, 0.5)
            else:
                agent.behavior.execute("Zero")

            cmd = agent.world.robot.get_command()
            self.comm_threads[aid].send_command(cmd)

        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        ball_pos = self.agents["agent_0"].world.ball.loc
        dists = {}

        for aid, comm_thread in self.comm_threads.items():
            ob = comm_thread.get_observation()
            if ob is None:
                ob = np.zeros(70, dtype=np.float32)
            obs[aid] = ob
            dists[aid] = np.linalg.norm(self.agents[aid].world.robot.loc[:2] - ball_pos[:2])

        closest = min(dists, key=dists.get)

        for aid, agent in self.agents.items():
            dist = dists[aid]
            terminated = agent.world.robot.cheat_abs_pos[2] < 0.3
            truncated = self.step_count >= self.max_steps

            terminateds[aid] = terminated
            truncateds[aid] = truncated

            if aid == closest:
                reward = -dist
                if dist < 0.8:
                    reward += 5.0
                    terminateds[aid] = True
            else:
                reward = -dist * 0.1

            rewards[aid] = reward
            infos[aid] = {"distance_to_ball": dist}

        terminateds["__all__"] = any(terminateds.values())
        truncateds["__all__"] = any(truncateds.values())

        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        for comm_thread in self.comm_threads.values():
            comm_thread.stop()
        for agent in self.agents.values():
            agent.terminate()
        Draw.clear_all()


def env_creator(config):
    worker_id = config.get("worker_index", 0)
    offset = worker_id * 10
    return MultiAgentSharedWorld({
        "ip": config["ip"],
        "robot_type": config["robot_type"],
        "server_p": config["server_p"] + offset,
        "monitor_p": config["monitor_p"] + offset,
    })


if __name__ == "__main__":
    ray.init()

    obs_space = gym.spaces.Box(low=-10, high=10, shape=(70,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    num_envs = 1

    register_env("MultiAgentSharedWorld", env_creator)

    config = (
        PPOConfig()
        .environment(
            env="MultiAgentSharedWorld",
            env_config={
                "ip": "127.0.0.1",
                "robot_type": 0,
                "server_p": 3100,
                "monitor_p": 3200,
            },
        )
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id: "shared_policy",
        )
        .framework("torch")
        .training(
            train_batch_size_per_learner=4000,
            minibatch_size=128,
            num_epochs=10,
            lr=3e-4,
        )
        .env_runners(
            num_env_runners=num_envs,
            num_envs_per_env_runner=1,
        )
        .resources(
            num_gpus=0,
        )
    )

    algo = config.build_algo()

    try:
        for i in range(200):
            result = algo.train()
            print(f"Iter: {i+1}, reward_mean: {result['episode_reward_mean']}")
    finally:
        algo.stop()
