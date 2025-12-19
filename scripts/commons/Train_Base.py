from datetime import datetime, timedelta
from itertools import count
from os import listdir
from os.path import isdir, join, isfile
from scripts.commons.UI import UI
from shutil import copy
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from typing import Callable
from world.World import World
from xml.dom import minidom
import numpy as np
import os, time, math, csv, select, sys
import pickle
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class SuccessLoggerCallback(BaseCallback):
    def __init__(self, rolling_window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.rolling_window = rolling_window
        self.successes = []
        self._ep_returns = None
        self._ep_lengths = None
        self._finished_returns = []
        self._finished_lengths = []
        self._last_rollout_idx = 0

        # ✅ per-iteration accumulators (include unfinished eps)
        self._rollout_reward_sum = 0.0
        self._rollout_steps = 0

        # NEW: track sum of squares to compute std for step rewards this iteration
        self._rollout_reward_sumsq = 0.0  # NEW

    def _ensure_buffers(self):
        n_envs = getattr(self.training_env, "num_envs", 1)
        if self._ep_returns is None or self._ep_returns.shape[0] != n_envs:
            self._ep_returns = np.zeros(n_envs, dtype=np.float32)
            self._ep_lengths = np.zeros(n_envs, dtype=np.int32)

    def _on_training_start(self) -> None:
        self._ensure_buffers()
        self._rollout_reward_sum = 0.0
        self._rollout_steps = 0
        self._rollout_reward_sumsq = 0.0  # NEW

    def _on_step(self) -> bool:
        self._ensure_buffers()

        r = np.atleast_1d(self.locals["rewards"]).astype(np.float32)
        d = np.atleast_1d(self.locals["dones"]).astype(bool)
        infos = self.locals.get("infos", [])

        # ongoing episode accumulators
        self._ep_returns += r
        self._ep_lengths += 1

        # include ALL rewards this iteration (finished + unfinished)
        self._rollout_reward_sum += float(r.sum())
        self._rollout_reward_sumsq += float(np.square(r).sum())  # NEW
        self._rollout_steps += int(r.size)

        for i, done in enumerate(d):
            if done:
                self._finished_returns.append(float(self._ep_returns[i]))
                self._finished_lengths.append(int(self._ep_lengths[i]))
                if i < len(infos) and "success" in infos[i]:
                    self.successes.append(int(infos[i]["success"]))
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
        return True

    def _on_rollout_end(self) -> None:
        # episodes that finished during THIS rollout
        new_returns = self._finished_returns[self._last_rollout_idx:]
        new_lengths = self._finished_lengths[self._last_rollout_idx:]
        self._last_rollout_idx = len(self._finished_returns)

        # log totals for this iteration
        total_finished_this = float(np.sum(new_returns)) if len(new_returns) else 0.0
        self.logger.record("rollout/reward_sum_all", self._rollout_reward_sum)
        self.logger.record("rollout/steps_this_iter", self._rollout_steps)
        self.logger.record("rollout/reward_sum_finished_eps", total_finished_this)

        # NEW: per-iteration step-reward std (all steps this rollout, finished + unfinished)
        if self._rollout_steps > 0:
            mean = self._rollout_reward_sum / self._rollout_steps
            mean_sq = self._rollout_reward_sumsq / self._rollout_steps
            var = max(0.0, mean_sq - mean * mean)
            self.logger.record("rollout/reward_mean_all", mean)          # NEW (optional)
            self.logger.record("rollout/reward_std_all", np.sqrt(var))   # NEW

        # existing per-iteration means for finished episodes
        if new_returns:
            self.logger.record("rollout/ep_rew_mean_this_rollout", float(np.mean(new_returns)))
            self.logger.record("rollout/ep_len_mean_this_rollout", float(np.mean(new_lengths)))
            self.logger.record("rollout/episodes_finished", len(new_returns))

            # NEW: per-iteration std for finished episodes
            self.logger.record("rollout/ep_rew_std_this_rollout", float(np.std(new_returns, ddof=0)))  # NEW
            self.logger.record("rollout/ep_len_std_this_rollout", float(np.std(new_lengths, ddof=0)))  # NEW

        # rolling means across training
        if self._finished_returns:
            rw = self.rolling_window
            last_r = self._finished_returns[-rw:]
            last_l = self._finished_lengths[-rw:]
            self.logger.record("rollout/ep_rew_mean_custom", float(np.mean(last_r)))
            self.logger.record("rollout/ep_len_mean_custom", float(np.mean(last_l)))
            # NEW: rolling std across last N finished episodes
            self.logger.record("rollout/ep_rew_std_custom", float(np.std(last_r, ddof=0)))  # NEW
            self.logger.record("rollout/ep_len_std_custom", float(np.std(last_l, ddof=0)))  # NEW

        if self.successes:
            self.logger.record("rollout/success_rate", sum(self.successes) / len(self.successes))
            self.successes.clear()

        # reset per-iteration accumulators
        self._rollout_reward_sum = 0.0
        self._rollout_reward_sumsq = 0.0  # NEW
        self._rollout_steps = 0


# class SuccessLoggerCallback(BaseCallback):
#     def __init__(self, rolling_window: int = 100, verbose: int = 0):
#         super().__init__(verbose)
#         self.rolling_window = rolling_window
#         self.successes = []
#         self._ep_returns = None
#         self._ep_lengths = None
#         self._finished_returns = []
#         self._finished_lengths = []
#         self._last_rollout_idx = 0

#         # ✅ per-iteration accumulators (include unfinished eps)
#         self._rollout_reward_sum = 0.0
#         self._rollout_steps = 0

#     def _ensure_buffers(self):
#         n_envs = getattr(self.training_env, "num_envs", 1)
#         if self._ep_returns is None or self._ep_returns.shape[0] != n_envs:
#             self._ep_returns = np.zeros(n_envs, dtype=np.float32)
#             self._ep_lengths = np.zeros(n_envs, dtype=np.int32)

#     def _on_training_start(self) -> None:
#         self._ensure_buffers()
#         self._rollout_reward_sum = 0.0
#         self._rollout_steps = 0

#     def _on_step(self) -> bool:
#         self._ensure_buffers()

#         r = np.atleast_1d(self.locals["rewards"]).astype(np.float32)
#         d = np.atleast_1d(self.locals["dones"]).astype(bool)
#         infos = self.locals.get("infos", [])

#         # ongoing episode accumulators
#         self._ep_returns += r
#         self._ep_lengths += 1

#         # ✅ include ALL rewards this iteration (finished + unfinished)
#         self._rollout_reward_sum += float(r.sum())
#         self._rollout_steps += int(r.size)

#         for i, done in enumerate(d):
#             if done:
#                 self._finished_returns.append(float(self._ep_returns[i]))
#                 self._finished_lengths.append(int(self._ep_lengths[i]))
#                 if i < len(infos) and "success" in infos[i]:
#                     self.successes.append(int(infos[i]["success"]))
#                 self._ep_returns[i] = 0.0
#                 self._ep_lengths[i] = 0
#         return True

#     def _on_rollout_end(self) -> None:
#         # episodes that finished during THIS rollout
#         new_returns = self._finished_returns[self._last_rollout_idx:]
#         new_lengths = self._finished_lengths[self._last_rollout_idx:]
#         self._last_rollout_idx = len(self._finished_returns)

#         # log totals for this iteration
#         total_finished_this = float(np.sum(new_returns)) if len(new_returns) else 0.0
#         self.logger.record("rollout/reward_sum_all", self._rollout_reward_sum)
#         self.logger.record("rollout/steps_this_iter", self._rollout_steps)
#         self.logger.record("rollout/reward_sum_finished_eps", total_finished_this)

#         # existing per-iteration means
#         if new_returns:
#             self.logger.record("rollout/ep_rew_mean_this_rollout", float(np.mean(new_returns)))
#             self.logger.record("rollout/ep_len_mean_this_rollout", float(np.mean(new_lengths)))
#             self.logger.record("rollout/episodes_finished", len(new_returns))

#         # rolling means across training
#         if self._finished_returns:
#             rw = self.rolling_window
#             self.logger.record("rollout/ep_rew_mean_custom", float(np.mean(self._finished_returns[-rw:])))
#             self.logger.record("rollout/ep_len_mean_custom", float(np.mean(self._finished_lengths[-rw:])))

#         if self.successes:
#             self.logger.record("rollout/success_rate", sum(self.successes) / len(self.successes))
#             self.successes.clear()

#         # reset per-iteration accumulators
#         self._rollout_reward_sum = 0.0
#         self._rollout_steps = 0


# class SuccessLoggerCallback(BaseCallback):
#     def __init__(self, rolling_window: int = 100, verbose: int = 0):
#         super().__init__(verbose)
#         self.rolling_window = rolling_window

#         # success flags collected during the current rollout
#         self.successes = []

#         # per-env running episode stats
#         self._ep_returns = None
#         self._ep_lengths = None

#         # all finished episodes (since training start)
#         self._finished_returns = []
#         self._finished_lengths = []

#         # index cursor: start of "new" finished episodes since last rollout end
#         self._last_rollout_idx = 0

#     def _ensure_buffers(self):
#         """Init/resize per-env buffers once we know num_envs."""
#         n_envs = getattr(self.training_env, "num_envs", 1)
#         if self._ep_returns is None or self._ep_returns.shape[0] != n_envs:
#             self._ep_returns = np.zeros(n_envs, dtype=np.float32)
#             self._ep_lengths = np.zeros(n_envs, dtype=np.int32)

#     def _on_training_start(self) -> None:
#         self._ensure_buffers()

#     def _on_step(self) -> bool:
#         self._ensure_buffers()

#         # rewards/dones can be scalars for single env; make them 1-D
#         r = np.atleast_1d(self.locals["rewards"]).astype(np.float32)  # (n_envs,)
#         d = np.atleast_1d(self.locals["dones"]).astype(bool)          # (n_envs,)
#         infos = self.locals.get("infos", [])

#         # accumulate ongoing episode stats
#         self._ep_returns += r
#         self._ep_lengths += 1

#         # flush finished episodes at this step
#         for i, done in enumerate(d):
#             if done:
#                 # store episode totals
#                 self._finished_returns.append(float(self._ep_returns[i]))
#                 self._finished_lengths.append(int(self._ep_lengths[i]))
#                 # record success flag if provided
#                 if i < len(infos) and "success" in infos[i]:
#                     self.successes.append(int(infos[i]["success"]))
#                 # reset per-env accumulators
#                 self._ep_returns[i] = 0.0
#                 self._ep_lengths[i] = 0

#         return True

#     def _on_rollout_end(self) -> None:
#         """Log per-rollout means and rolling means; then clear rollout-scoped data."""
#         # episodes that finished during THIS rollout window
#         new_returns = self._finished_returns[self._last_rollout_idx:]
#         new_lengths = self._finished_lengths[self._last_rollout_idx:]
#         self._last_rollout_idx = len(self._finished_returns)

#         if new_returns:
#             mean_r_this = float(np.mean(new_returns))
#             mean_l_this = float(np.mean(new_lengths))
#             self.logger.record("rollout/ep_rew_mean_this_rollout", mean_r_this)
#             self.logger.record("rollout/ep_len_mean_this_rollout", mean_l_this)
#             self.logger.record("rollout/episodes_finished", len(new_returns))

#         # rolling mean over the last N finished episodes across ALL training so far
#         if self._finished_returns:
#             rw = self.rolling_window
#             mean_r_rolling = float(np.mean(self._finished_returns[-rw:]))
#             mean_l_rolling = float(np.mean(self._finished_lengths[-rw:]))
#             self.logger.record("rollout/ep_rew_mean_custom", mean_r_rolling)
#             self.logger.record("rollout/ep_len_mean_custom", mean_l_rolling)

#         # success rate for episodes that ended in THIS rollout
#         if self.successes:
#             self.logger.record("rollout/success_rate", sum(self.successes) / len(self.successes))
#             self.successes.clear()




class Train_Base():
    def __init__(self, script) -> None:
        '''
        When training with multiple environments (multiprocessing):
            The server port is incremented as follows:
                self.server_p, self.server_p+1, self.server_p+2, ...
            We add +1000 to the initial monitor port, so than we can have more than 100 environments:
                self.monitor_p+1000, self.monitor_p+1001, self.monitor_p+1002, ...
        When testing we use self.server_p and self.monitor_p
        '''

        args = script.args
        self.script = script
        self.ip = args.i
        self.server_p = args.p              # (initial) server port
        self.monitor_p = args.m             # monitor port when testing
        self.monitor_p_1000 = args.m + 1000 # initial monitor port when training
        self.robot_type = args.r
        self.team = args.t
        self.uniform = args.u
        self.cf_last_time = 0
        self.cf_delay = 0
        self.cf_target_period = World.STEPTIME # target simulation speed while testing (default: real-time)

    @staticmethod
    def prompt_user_for_model():

        gyms_logs_path = "./scripts/gyms/logs/"
        folders = [f for f in listdir(gyms_logs_path) if isdir(join(gyms_logs_path, f))]
        folders.sort(key=lambda f: os.path.getmtime(join(gyms_logs_path, f)), reverse=True) # sort by modification date

        while True:
            try:
                folder_name = UI.print_list(folders,prompt="Choose folder (ctrl+c to return): ")[1]
            except KeyboardInterrupt:
                print()
                return None # ctrl+c

            folder_dir = os.path.join(gyms_logs_path, folder_name)
            models = [m[:-4] for m in listdir(folder_dir) if isfile(join(folder_dir, m)) and m.endswith(".zip")]

            if not models:
                print("The chosen folder does not contain any .zip file!")
                continue

            models.sort(key=lambda m: os.path.getmtime(join(folder_dir, m+".zip")), reverse=True) # sort by modification date
            
            try:
                model_name = UI.print_list(models,prompt="Choose model (ctrl+c to return): ")[1]
                break
            except KeyboardInterrupt:
                print()

        return {"folder_dir":folder_dir, "folder_name":folder_name, "model_file":os.path.join(folder_dir, model_name+".zip")}


    def control_fps(self, read_input = False):
        ''' Add delay to control simulation speed '''

        if read_input:
            speed = input()
            if speed == '':
                self.cf_target_period = 0
                print(f"Changed simulation speed to MAX")
            else:
                if speed == '0':
                    inp = input("Paused. Set new speed or '' to use previous speed:")
                    if inp != '':
                        speed = inp   

                try:
                    speed = int(speed)
                    assert speed >= 0
                    self.cf_target_period = World.STEPTIME * 100 / speed
                    print(f"Changed simulation speed to {speed}%")
                except:
                    print("""Train_Base.py: 
    Error: To control the simulation speed, enter a non-negative integer.
    To disable this control module, use test_model(..., enable_FPS_control=False) in your gym environment.""")

        now = time.time()
        period = now - self.cf_last_time
        self.cf_last_time = now
        self.cf_delay += (self.cf_target_period - period)*0.9
        if self.cf_delay > 0:
            time.sleep(self.cf_delay)
        else:
            self.cf_delay = 0


    def test_model(self, model:BaseAlgorithm, env, log_path:str=None, model_path:str=None, max_episodes=0, enable_FPS_control=True, verbose=1):
        '''
        Test model and log results

        Parameters
        ----------
        model : BaseAlgorithm
            Trained model 
        env : Env
            Gym-like environment
        log_path : str
            Folder where statistics file is saved, default is `None` (no file is saved)
        model_path : str
            Folder where it reads evaluations.npz to plot it and create evaluations.csv, default is `None` (no plot, no csv)
        max_episodes : int
            Run tests for this number of episodes
            Default is 0 (run until user aborts)
        verbose : int
            0 - no output (except if enable_FPS_control=True)
            1 - print episode statistics
        '''

        if model_path is not None:
            assert os.path.isdir(model_path), f"{model_path} is not a valid path"
            self.display_evaluations(model_path)

        if log_path is not None:
            assert os.path.isdir(log_path), f"{log_path} is not a valid path"

            # If file already exists, don't overwrite
            if os.path.isfile(log_path + "/test.csv"):
                for i in range(1000):
                    p = f"{log_path}/test_{i:03}.csv"
                    if not os.path.isfile(p):
                        log_path = p
                        break
            else:
                log_path += "/test.csv"
            
            with open(log_path, 'w') as f:
                f.write("reward,ep. length,rew. cumulative avg., ep. len. cumulative avg.\n")
            print("Train statistics are saved to:", log_path)

        if enable_FPS_control: # control simulation speed (using non blocking user input)
            print("\nThe simulation speed can be changed by sending a non-negative integer\n"
                  "(e.g. '50' sets speed to 50%, '0' pauses the simulation, '' sets speed to MAX)\n")

        ep_reward = 0
        ep_length = 0
        rewards_sum = 0
        reward_min = math.inf
        reward_max = -math.inf
        ep_lengths_sum = 0
        ep_no = 0
        robot_positions = []
        ball_positions  = []
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            robot_positions.append(info["robot_pos"])
            ball_positions.append(info["ball_pos"])
            ep_reward += reward
            ep_length += 1

            if enable_FPS_control: # control simulation speed (using non blocking user input)
                self.control_fps(select.select([sys.stdin], [], [], 0)[0]) 

            if done:
            #    # --- PLOT TRAJECTORY (NEW) ---
            #     if robot_positions and ball_positions:
            #         robot_positions = robot_positions[20:]
            #         ball_positions = ball_positions[20:]
            #         r_x, r_y = zip(*robot_positions)
            #         b_x, b_y = zip(*ball_positions)

            #         plt.figure(figsize=(6, 6))

            #         # Robot trajectory (blue)
            #         plt.plot(r_x, r_y, color='blue', label="Robot")
            #         plt.scatter(r_x[0], r_y[0],
            #                     marker='o', color='blue', edgecolor='black', s=100, label="Robot Start")

            #         # Ball trajectory (orange, dashed)
            #         plt.plot(b_x, b_y, linestyle='--', color='orange', label="Ball")
            #         plt.scatter(b_x[0], b_y[0],
            #                     marker='o', color='orange', edgecolor='black', s=100, label="Ball Start")

            #         # Axis style
            #         plt.gca().invert_yaxis()   # Soccer-sim Y axis is usually inverted
            #         plt.xlabel("X position")
            #         plt.ylabel("Y position")
            #         plt.title(f"Episode {ep_no+1} trajectories")
            #         plt.legend()
            #         plt.grid(True)
            #         plt.tight_layout()

            #         # choose where to save
            #         img_path = os.path.join(os.path.dirname(log_path) if log_path else ".", 
            #                                 f"trajectory_ep_{ep_no+1}.png")
            #         plt.savefig(img_path)
            #         plt.close()
            #         print(f"\nSaved trajectory to {img_path}")




                
                # if ep_no == 5:
                #     if robot_positions and ball_positions:
                #         r_x, r_y = zip(*robot_positions)
                #         b_x, b_y = zip(*ball_positions)

                #         t_vals = np.linspace(0, 1, len(r_x))  # normalized time steps

                #         plt.figure(figsize=(7, 6))

                #         # Color-coded scatter
                #         plt.scatter(r_x, r_y, c=t_vals, cmap='viridis', marker='x', label='Robot', s=35)
                #         plt.scatter(b_x, b_y, c=t_vals, cmap='plasma', marker='o', label='Ball', s=25)

                #         # Optional faint line connecting trajectory
                #         plt.plot(r_x, r_y, color='grey', linewidth=0.5, alpha=0.4)
                #         plt.plot(b_x, b_y, color='grey', linewidth=0.5, alpha=0.4)

                #         # Start and end points
                #         plt.scatter(r_x[0], r_y[0], marker='^', color='green', s=80, label='Robot Start')
                #         plt.scatter(r_x[-1], r_y[-1], marker='s', color='red', s=80, label='Robot End')
                #         plt.scatter(b_x[0], b_y[0], marker='^', color='green', s=80, label='Ball Start')
                #         plt.scatter(b_x[-1], b_y[-1], marker='s', color='red', s=80, label='Ball End')

                #         plt.xlabel("X position")
                #         plt.ylabel("Y position")
                #         plt.title(f"Episode {ep_no+1} Trajectory (Robot vs Ball)")
                #         plt.legend()
                #         plt.grid(True)
                #         plt.colorbar(label="Normalized Time →")
                #         plt.gca().invert_yaxis()
                #         plt.gca().set_aspect("equal")
                #         plt.tight_layout()

                #         img_path = os.path.join(log_path.rsplit("/",1)[0] if log_path else ".", 
                #                                 f"trajectory_ep_{ep_no+1}.png")
                #         plt.savefig(img_path)
                #         plt.close()
                #         print(f"\nSaved trajectory to {img_path}")

                robot_positions = []
                ball_positions  = []
                obs = env.reset()
                rewards_sum += ep_reward
                ep_lengths_sum += ep_length
                reward_max = max(ep_reward, reward_max)
                reward_min = min(ep_reward, reward_min)
                ep_no += 1
                avg_ep_lengths = ep_lengths_sum/ep_no
                avg_rewards = rewards_sum/ep_no

                if verbose > 0:
                    print(  f"\rEpisode: {ep_no:<3}  Ep.Length: {ep_length:<4.0f}  Reward: {ep_reward:<6.2f}                                                             \n",
                        end=f"--AVERAGE--   Ep.Length: {avg_ep_lengths:<4.0f}  Reward: {avg_rewards:<6.2f}  (Min: {reward_min:<6.2f}  Max: {reward_max:<6.2f})", flush=True)
                
                if log_path is not None:
                    with open(log_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([ep_reward, ep_length, avg_rewards, avg_ep_lengths])
                
                if ep_no == max_episodes:
                    return

                ep_reward = 0
                ep_length = 0

    def learn_model(self, model:BaseAlgorithm, total_steps:int, path:str, eval_env=None, eval_freq=None, eval_eps=5, save_freq=None, backup_env_file=None, export_name=None):
        '''
        Learn Model for a specific number of time steps

        Parameters
        ----------
        model : BaseAlgorithm
            Model to train
        total_steps : int
            The total number of samples (env steps) to train on
        path : str
            Path where the trained model is saved
            If the path already exists, an incrementing number suffix is added
        eval_env : Env
            Environment to periodically test the model
            Default is None (no periodical evaluation)
        eval_freq : int
            Evaluate the agent every X steps
            Default is None (no periodical evaluation)
        eval_eps : int
            Evaluate the agent for X episodes (both eval_env and eval_freq must be defined)
            Default is 5
        save_freq : int
            Saves model at every X steps
            Default is None (no periodical checkpoint)
        backup_gym_file : str
            Generates backup of environment file in model's folder
            Default is None (no backup)
        export_name : str
            If export_name and save_freq are defined, a model is exported every X steps
            Default is None (no export)

        Returns
        -------
        model_path : str
            Directory where model was actually saved (considering incremental suffix)

        Notes
        -----
        If `eval_env` and `eval_freq` were specified:
            - The policy will be evaluated in `eval_env` every `eval_freq` steps
            - Evaluation results will be saved in `path` and shown at the end of training
            - Every time the results improve, the model is saved
        '''

        start = time.time()
        start_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # If path already exists, add suffix to avoid overwriting
        if os.path.isdir(path):
            for i in count():
                p = path.rstrip("/")+f'_{i:03}/'
                if not os.path.isdir(p):
                    path = p
                    break
        os.makedirs(path)

        # Backup environment file
        if backup_env_file is not None:
            backup_file = os.path.join(path, os.path.basename(backup_env_file))
            copy(backup_env_file, backup_file)

        evaluate = bool(eval_env is not None and eval_freq is not None)

        # Create evaluation callback
        eval_callback = None if not evaluate else EvalCallback(eval_env, n_eval_episodes=eval_eps, eval_freq=eval_freq, log_path=path,
                                                               best_model_save_path=path, deterministic=True, render=False)

        # Create custom callback to display evaluations
        custom_callback = None if not evaluate else Cyclic_Callback(eval_freq, lambda:self.display_evaluations(path,True))

        # Create checkpoint callback
        checkpoint_callback = None if save_freq is None else CheckpointCallback(save_freq=save_freq, save_path=path, name_prefix="model", verbose=1)

        # Create custom callback to export checkpoint models
        export_callback = None if save_freq is None or export_name is None else Export_Callback(save_freq, path, export_name)

        #
        success_logger = SuccessLoggerCallback()

        callbacks = CallbackList([c for c in [success_logger, eval_callback, custom_callback, checkpoint_callback, export_callback] if c is not None])
        #

        from stable_baselines3.common.logger import configure

        log_path = path  # Same as your model saving path
        new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])  # 👈 includes terminal logging
        model.set_logger(new_logger)


        # callbacks = CallbackList([c for c in [eval_callback, custom_callback, checkpoint_callback, export_callback] if c is not None])

        model.learn( total_timesteps=total_steps, callback=callbacks )
        model.save( os.path.join(path, "last_model") )

        # Display evaluations if they exist
        if evaluate:
            self.display_evaluations(path)

        # Display timestamps + Model path
        end_date = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        duration = timedelta(seconds=int(time.time()-start))
        print(f"Train start:     {start_date}")
        print(f"Train end:       {end_date}")
        print(f"Train duration:  {duration}")
        print(f"Model path:      {path}")
        
        # Append timestamps to backup environment file
        if backup_env_file is not None:
            with open(backup_file, 'a') as f:
                f.write(f"\n# Train start:    {start_date}\n")
                f.write(  f"# Train end:      {end_date}\n")
                f.write(  f"# Train duration: {duration}")

        return path

    def display_evaluations(self, path, save_csv=False):

        eval_npz = os.path.join(path, "evaluations.npz")

        if not os.path.isfile(eval_npz):
            return

        console_width = 80
        console_height = 18
        symb_x = "\u2022"
        symb_o = "\u007c"
        symb_xo = "\u237f"

        with np.load(eval_npz) as data:
            time_steps = data["timesteps"]
            results_raw = np.mean(data["results"],axis=1)
            ep_lengths_raw = np.mean(data["ep_lengths"],axis=1)
        sample_no = len(results_raw)

        xvals = np.linspace(0, sample_no-1, 80)
        results    = np.interp(xvals, range(sample_no), results_raw)
        ep_lengths = np.interp(xvals, range(sample_no), ep_lengths_raw)

        results_limits    = np.min(results),    np.max(results)
        ep_lengths_limits = np.min(ep_lengths), np.max(ep_lengths)

        results_discrete    = np.digitize(results,    np.linspace(results_limits[0]-1e-5, results_limits[1]+1e-5,    console_height+1))-1
        ep_lengths_discrete = np.digitize(ep_lengths, np.linspace(0,                      ep_lengths_limits[1]+1e-5, console_height+1))-1

        matrix = np.zeros((console_height, console_width, 2), int)
        matrix[results_discrete[0]   ][0][0] = 1    # draw 1st column
        matrix[ep_lengths_discrete[0]][0][1] = 1    # draw 1st column
        rng = [[results_discrete[0], results_discrete[0]], [ep_lengths_discrete[0], ep_lengths_discrete[0]]]

        # Create continuous line for both plots
        for k in range(2):
            for i in range(1,console_width):
                x = [results_discrete, ep_lengths_discrete][k][i]
                if x > rng[k][1]:
                    rng[k] = [rng[k][1]+1, x]
                elif x < rng[k][0]:
                    rng[k] = [x, rng[k][0]-1]
                else:
                    rng[k] = [x,x]
                for j in range(rng[k][0],rng[k][1]+1):
                    matrix[j][i][k] = 1

        print(f'{"-"*console_width}')
        for l in reversed(range(console_height)):
            for c in range(console_width):
                if   np.all(matrix[l][c] == 0): print(end=" ")
                elif np.all(matrix[l][c] == 1): print(end=symb_xo)
                elif matrix[l][c][0] == 1:      print(end=symb_x)
                else:                           print(end=symb_o)
            print()
        print(f'{"-"*console_width}')
        print(f"({symb_x})-reward          min:{results_limits[0]:11.2f}    max:{results_limits[1]:11.2f}")
        print(f"({symb_o})-ep. length      min:{ep_lengths_limits[0]:11.0f}    max:{ep_lengths_limits[1]:11.0f}    {time_steps[-1]/1000:15.0f}k steps")
        print(f'{"-"*console_width}')

        # save CSV
        if save_csv:
            eval_csv = os.path.join(path, "evaluations.csv")
            with open(eval_csv, 'a+') as f:
                writer = csv.writer(f)
                if sample_no == 1:
                    writer.writerow(["time_steps", "reward ep.", "length"])
                writer.writerow([time_steps[-1],results_raw[-1],ep_lengths_raw[-1]])


    def generate_slot_behavior(self, path, slots, auto_head:bool, XML_name):
        '''
        Function that generates the XML file for the optimized slot behavior, overwriting previous files
        '''

        file = os.path.join( path, XML_name )

        # create the file structure
        auto_head = '1' if auto_head else '0'
        EL_behavior = ET.Element('behavior',{'description':'Add description to XML file', "auto_head":auto_head})

        for i,s in enumerate(slots):
            EL_slot = ET.SubElement(EL_behavior, 'slot', {'delta':str(s[0]/1000)})
            for j in s[1]: # go through all joint indices
                ET.SubElement(EL_slot, 'move', {'id':str(j), 'angle':str(s[2][j])})

        # create XML file
        xml_rough = ET.tostring( EL_behavior, 'utf-8' )
        xml_pretty = minidom.parseString(xml_rough).toprettyxml(indent="    ")
        with open(file, "w") as x:
            x.write(xml_pretty)
        
        print(file, "was created!")

    @staticmethod
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        '''
        Linear learning rate schedule

        Parameters
        ----------
        initial_value : float
            Initial learning rate
        
        Returns
        -------
        schedule : Callable[[float], float]
            schedule that computes current learning rate depending on remaining progress
        '''
        def func(progress_remaining: float) -> float:
            '''
            Compute learning rate according to current progress

            Parameters
            ----------
            progress_remaining : float
                Progress will decrease from 1 (beginning) to 0
            
            Returns
            -------
            learning_rate : float
                Learning rate according to current progress
            '''
            return progress_remaining * initial_value

        return func

    @staticmethod
    def export_model(input_file, output_file, add_sufix=True):
        '''
        Export model weights to binary file

        Parameters
        ----------
        input_file : str
            Input file, compatible with algorithm
        output_file : str
            Output file, including directory
        add_sufix : bool
            If true, a suffix is appended to the file name: output_file + "_{index}.pkl"
        '''

        # If file already exists, don't overwrite
        if add_sufix:
            for i in count():
                f = f"{output_file}_{i:03}.pkl"
                if not os.path.isfile(f):
                    output_file = f
                    break
        
        model = PPO.load(input_file)
        weights = model.policy.state_dict() # dictionary containing network layers

        w = lambda name : weights[name].detach().cpu().numpy() # extract weights from policy

        var_list = []
        for i in count(0,2): # add hidden layers (step=2 because that's how SB3 works)
            if f"mlp_extractor.policy_net.{i}.bias" not in weights:
                break
            var_list.append([w(f"mlp_extractor.policy_net.{i}.bias"), w(f"mlp_extractor.policy_net.{i}.weight"), "tanh"])

        var_list.append( [w("action_net.bias"), w("action_net.weight"), "none"] ) # add final layer
        
        with open(output_file,"wb") as f:
            pickle.dump(var_list, f, protocol=4) # protocol 4 is backward compatible with Python 3.4



class Cyclic_Callback(BaseCallback):
    ''' Stable baselines custom callback '''
    def __init__(self, freq, function):
        super(Cyclic_Callback, self).__init__(1)
        self.freq = freq
        self.function = function

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.function()
        return True # If the callback returns False, training is aborted early

class Export_Callback(BaseCallback):
    ''' Stable baselines custom callback '''
    def __init__(self, freq, load_path, export_name):
        super(Export_Callback, self).__init__(1)
        self.freq = freq
        self.load_path = load_path
        self.export_name = export_name

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            path = os.path.join(self.load_path, f"model_{self.num_timesteps}_steps.zip")
            Train_Base.export_model(path, f"./scripts/gyms/export/{self.export_name}")
        return True # If the callback returns False, training is aborted early