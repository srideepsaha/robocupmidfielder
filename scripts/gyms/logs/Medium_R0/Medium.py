import gym
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from agent.Base_Agent import Base_Agent
from world.commons.Draw import Draw
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep


class InterceptionEnv(gym.Env):
    def __init__(self, ip, server_p, monitor_p, robot_type, enable_draw, ball_speed = 4.0) -> None:
        super().__init__()

        self.robot_type = robot_type

        # Use Base_Agent directly (like Basic_Run)
        self.player = Base_Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)

        self.step_counter = 0
        self.max_steps = 500 
        self.last_distance = np.inf
        self.ball_speed = ball_speed

        self.observation_space = gym.spaces.Box(
            low=np.array([-10.0, -10.0, 0.0, -np.pi, -5.0, -5.0]),
            high=np.array([10.0, 10.0, 10.0, np.pi, 5.0, 5.0]),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(5)
        self.player.scom.unofficial_set_play_mode("PlayOn")

    def teleport_ball_fixed(self):
        x_pos = 0.0
        y_pos = random.uniform(-1.5, 1.5)
        speed_y = random.uniform(1.5 - y_pos, -1.5 - y_pos)
        #self.player.scom.unofficial_move_ball((x_pos, y_pos, 0.0), (0.0, 0.0, 0.0))
        self.player.scom.unofficial_move_ball((x_pos, y_pos, 0.0), (-self.ball_speed, speed_y, 0))
        #self.player.scom.unofficial_move_ball((1.0, 0.0, 0.042), (0.0, 0.0, 0.0))
        # for _ in range(2):
        #     self.player.behavior.execute("Zero")
        #     r = self.player.world.robot
        #     self.player.scom.commit_and_send(r.get_command())
        #     self.player.scom.receive()

    def observe(self):
        w = self.player.world
        robot_pos = w.robot.loc_head_position[:2]
        ball_pos = w.ball_abs_pos[:2]
        rel_ball_pos = w.ball_rel_torso_cart_pos[:2]
        init = [0.0,0.0]
        distance = np.linalg.norm(rel_ball_pos - init)
        ball_heading = np.arctan2(rel_ball_pos[1], rel_ball_pos[0])
        torso_orientation = np.deg2rad(w.robot.imu_torso_orientation)
        heading_diff = ball_heading - torso_orientation
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        ball_vel = w.ball_abs_vel[:2]
        obs = np.array([rel_ball_pos[0], rel_ball_pos[1], distance, ball_heading, ball_vel[0], ball_vel[1]], dtype=np.float32)
        return obs

    def sync(self):
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()

    def reset(self):
        self.last_distance = np.inf
        #self.teleport_ball_fixed()
        self.step_counter = 0
        r = self.player.world.robot

        for _ in range(3):
            self.player.scom.unofficial_beam((-3, 0, 0.5), 0)
            self.player.behavior.execute("Zero")
            self.sync()

        self.teleport_ball_fixed()

        return self.observe()

    def step(self, action):
        self.player.scom.unofficial_set_game_time(0)
        w = self.player.world

        if action == 0:
            target_relative = np.array([0.0, 5.0])
            speed = 0.5
        elif action == 1:
            target_relative = np.array([0.0, -5.0])
            speed = 0.5
        elif action == 3:
            target_relative = np.array([0.0, 5.0])
            speed = 0.1
        elif action == 4:
            target_relative = np.array([0.0, -5.0])
            speed = 0.1
        else:
            target_relative = np.array([0.0, 0.0])
            speed = 0.5

        self.player.behavior.execute(
            "Walk",
            target_relative,
            False,
            0.0,
            True,
            speed
        )

        self.sync()
        self.step_counter += 1

        obs = self.observe()
        rel_x, rel_y, distance, heading_diff, ball_vel_x, ball_vel_y = obs
        done = False
        distance_reward = -distance
        heading_reward = -abs(heading_diff)

        # Prioritize distance more, especially early
        reward = 0.6 * distance_reward + 0.4 * heading_reward

        if self.step_counter < 25:
            if action!=2:
                reward -= 0.2

        # Small time penalty to encourage quickness
        reward += -0.2

        # if action != 2:
        #     reward -= 0.1

        # NEXT add timestep penalty and do if both heading and distance decrease, what happens

        if self.step_counter%10 == 0:
            if distance > self.last_distance:
                reward -= 0.25
            else:
                self.last_distance = distance
                reward += 0.3

        # Smoother alignment bonus when getting close
        # if distance < 1.0:
        #     reward += (1.0 - distance) * 2  # smooth bonus

        # Success condition
        if distance <= 0.8 and abs(heading_diff) <= 0.03:
            if action == 2:
                reward += 5
                done = True
            else:
                reward -=2.5


        

        #reward = 0

        #reward = (4.33 - distance) * 0.15

        # if distance < self.last_distance:
        #     reward += 0.5
        # else:
        #     reward = -0.4
            
        # self.last_distance = distance

        # reward = max(0, 5 - distance)

        # After 10 steps use last distance to add a penalty if robot has moved in wrong direction
        # Can use angle

        # For final reward can also do based o nfinal distance to ball, like 5 - distance  

        if self.step_counter >= self.max_steps:
            if abs(heading_diff) > 0.03:
                reward -= 5  # penalty for failing
            done = True


        # if distance <= 3.05:
        #     reward += 25
        #     done = True

        # if self.step_counter >= self.max_steps:
        #     if distance > 3.05:
        #         reward-=10
        #     # if distance > 3.3:
        #     #     reward = 0
        #     # else:
        #     #     reward += 5
        #     done = True

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

        n_envs = 10
        n_steps_per_env = 512
        minibatch_size = 64
        total_steps_stage1 = 100_000
        total_steps_stage2 = 300_000
        learning_rate = 3e-4
        folder_name = f'Medium_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'
        print("Model path:", model_path)

        # ========== CURRICULUM LEARNING STAGE 1 ==========
        def init_env_stage1(i_env):
            def thunk():
                return InterceptionEnv(
                    self.ip,
                    self.server_p + i_env,
                    self.monitor_p_1000 + i_env,
                    self.robot_type,
                    enable_draw=False,
                    ball_speed=4.0
                )
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)
        env_stage1 = SubprocVecEnv([init_env_stage1(i) for i in range(n_envs)])
        eval_env_stage1 = SubprocVecEnv([init_env_stage1(n_envs)])

        model = PPO(
            "MlpPolicy",
            env=env_stage1,
            verbose=1,
            n_steps=n_steps_per_env,
            batch_size=minibatch_size,
            learning_rate=learning_rate,
            device="cpu"
        )

        model_path = self.learn_model(
            model,
            total_steps_stage1,
            model_path,
            eval_env=eval_env_stage1,
            eval_freq=n_steps_per_env * 20,
            save_freq=n_steps_per_env * 200,
            backup_env_file=__file__
        )

        env_stage1.close()
        eval_env_stage1.close()
        servers.kill()

        # ========== CURRICULUM LEARNING STAGE 2 ==========
        def init_env_stage2(i_env):
            def thunk():
                return InterceptionEnv(
                    self.ip,
                    self.server_p + i_env,
                    self.monitor_p_1000 + i_env,
                    self.robot_type,
                    enable_draw=False,
                    ball_speed=6.0
                )
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)
        env_stage2 = SubprocVecEnv([init_env_stage2(i) for i in range(n_envs)])
        eval_env_stage2 = SubprocVecEnv([init_env_stage2(n_envs)])

        model.set_env(env_stage2)

        model_path = self.learn_model(
            model,
            total_steps_stage2,
            model_path,
            eval_env=eval_env_stage2,
            eval_freq=n_steps_per_env * 20,
            save_freq=n_steps_per_env * 200,
            backup_env_file=__file__
        )

        env_stage2.close()
        eval_env_stage2.close()
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

# Train start:    22/06/2025 16:25:47
# Train end:      22/06/2025 16:28:30
# Train duration: 0:02:43