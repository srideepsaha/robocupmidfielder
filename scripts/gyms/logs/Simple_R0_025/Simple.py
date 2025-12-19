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
    def __init__(self, ip, server_p, monitor_p, robot_type, enable_draw) -> None:
        super().__init__()

        self.robot_type = robot_type

        # Use Base_Agent directly (like Basic_Run)
        self.player = Base_Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)

        self.step_counter = 0
        self.max_steps = 500 
        self.last_distance = np.inf
        self.previous_action = None

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
        self.player.scom.unofficial_move_ball((x_pos, y_pos, 0.0), (-4.0, speed_y, 0))
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

        for _ in range(25):
            self.player.scom.unofficial_beam((-3, 0, 0.5), 0)
            self.player.behavior.execute("Zero")
            self.sync()

        # beam player to ground
        self.player.scom.unofficial_beam((-3,0,r.beam_height),0) 
        r.joints_target_speed[0] = 0.01 # move head to trigger physics update (rcssserver3d bug when no joint is moving)
        self.sync()

        for _ in range(7): 
            self.player.behavior.execute("Zero")
            self.sync()

        self.teleport_ball_fixed()

        return self.observe()

    def step(self, action):
        success = False
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
        reward = 0.6 * distance_reward + 0.4 * heading_reward #Previously 0.6 0.4

        if self.step_counter < 25:
            if action!=2:
                reward -= 0.2

        # Small time penalty to encourage quickness
        reward += -0.4 #previously -0.2

        # if action != 2:
        #     reward -= 0.1

        # NEXT add timestep penalty and do if both heading and distance decrease, what happens

        if self.step_counter%10 == 0:
            if distance > self.last_distance:
                reward -= 0.25
            else:
                self.last_distance = distance
                reward += 0.3

        if distance <= 2.0:
            if abs(rel_y) > 0.15:
                reward -=0.25

        #####Add another condition when distance close, if starfe is fast penalize, so robot slows starfing near end

        # Smoother alignment bonus when getting close
        # if distance < 1.0:
        #     reward += (1.0 - distance) * 2  # smooth bonus
        # Success condition
        if distance <= 0.8 and abs(heading_diff) <= 0.03:
            if action == 2:
                reward += 5
                done = True
                success = True
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

        self.previous_action = action
        info = {
                    "robot_pos": w.robot.loc_torso_position[:2].copy(),
                    "ball_pos":  w.ball_cheat_abs_pos[:2].copy()
                }

        if done:
            info["success"] = success

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()


class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):
        #n_envs = min(16, os.cpu_count())
        n_envs = 10
        n_steps_per_env = 512
        minibatch_size = 64
        total_steps = 200000
        learning_rate = 3e-4
        folder_name = f'Simple_R{self.robot_type}'
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
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"], max_episodes=50)
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()
