import gym
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from agent.Agent import Agent
from world.commons.Draw import Draw
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep

class InterceptionEnv(gym.Env):
    def __init__(self, ip, server_p, monitor_p, robot_type, enable_draw) -> None:
        super().__init__()

        self.robot_type = robot_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0
        self.max_steps = 300

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -np.pi]),
            high=np.array([np.inf, np.pi]),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

    def teleport_ball_random(self):
        x_pos = 0.0
        y_pos = random.uniform(-1.5, 1.5)
        speed_y = random.uniform(1.5 - y_pos, -1.5 - y_pos)
        #self.player.scom.unofficial_move_ball((x_pos, y_pos, 0.0), (-6.0, speed_y, 0))
        self.player.scom.unofficial_move_ball((x_pos, y_pos, 0.0), (0, 0, 0))

    def observe(self):
        w = self.player.world
        robot_pos = w.robot.loc_head_position[:2]
        ball_pos = w.ball_abs_pos[:2]

        relative_pos = ball_pos - robot_pos
        distance = np.linalg.norm(relative_pos)
        angle_to_ball = np.arctan2(relative_pos[1], relative_pos[0])

        torso_orientation = np.deg2rad(w.robot.imu_torso_orientation)
        heading_diff = angle_to_ball - torso_orientation
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))

        obs = np.array([distance, heading_diff], dtype=np.float32)
        return obs

    def reset(self):
        self.step_counter = 0
        for _ in range(3):
            self.player.scom.unofficial_beam((-3, 0, 0.5), 0)
            self.player.behavior.execute("Zero")
            self.player.scom.commit_and_send(self.player.world.robot.get_command())
            self.player.scom.receive()

        for _ in range(3):
            self.player.behavior.execute("Zero")
            self.player.scom.commit_and_send(self.player.world.robot.get_command())
            self.player.scom.receive()

        self.teleport_ball_random()

        return self.observe()

    def step(self, action):
        w = self.player.world

        if action == 0:
            target_relative = np.array([0.0, 5.0])
        elif action == 1:
            target_relative = np.array([0.0, -5.0])
        else:
            target_relative = np.array([0.0, 0.0])

        self.player.behavior.execute(
            "Walk",
            target_relative,
            False,
            0.0,
            True,
            0.5
        )

        self.player.scom.commit_and_send(w.robot.get_command())
        self.player.scom.receive()

        self.step_counter += 1

        obs = self.observe()
        distance, heading_diff = obs

        distance_reward = -distance
        angle_reward = -abs(heading_diff)
        reward = distance_reward + 0.5 * angle_reward

        done = False
        if distance < 0.4:
            reward += 5
            done = True

        if self.step_counter >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()


class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):
        n_envs = min(16, os.cpu_count())
        n_steps_per_env = 512
        minibatch_size = 64
        total_steps = 10000000
        learning_rate = 3e-4
        folder_name = f'Interception_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'
        print("Model path:", model_path)

        def init_env(i_env):
            def thunk():
                return InterceptionEnv(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)

        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])

        try:
            if "model_file" in args:
                model = PPO.load(
                    args["model_file"], env=env, device="cpu",
                    n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate
                )
            else:
                model = PPO(
                    "MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env,
                    batch_size=minibatch_size, learning_rate=learning_rate, device="cpu"
                )

            model_path = self.learn_model(
                model, total_steps, model_path,
                eval_env=eval_env,
                eval_freq=n_steps_per_env * 20,
                save_freq=n_steps_per_env * 200,
                backup_env_file=__file__
            )
        except KeyboardInterrupt:
            sleep(1)
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return

        env.close()
        eval_env.close()
        servers.kill()

    def test(self, args):
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = InterceptionEnv(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)

        try:
            self.export_model(args["model_file"], args["model_file"] + ".pkl", False)
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()
