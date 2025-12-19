# RoboCup SPL Midfielder Interception (Reinforcement Learning)

## Summary
This codebase implements and evaluates reinforcement learning (RL) controllers for a RoboCup SPL-style **midfielder interception** task: a Nao robot (single-agent) or two robots (multi-agent) learn to **intercept a moving ball primarily by lateral sidestepping**. The work is trained and tested in simulation (SimSpark / RoboViz workflow) and includes practical notes for transferring the learned behavior toward real-robot constraints (Sim2Real).

## Problem setup (core idea)
- **Goal:** intercept the ball before it passes the robot / times out.
- **Primary control:** discrete **sidestep left / sidestep right / stand** (single-agent variant also uses slow/fast sidesteps).
- **Training loop:** episodic RL with shaped rewards that encourage closing distance, aligning heading, and avoiding unnecessary motion.

## Single-agent approach (Stable-Baselines3 PPO)
A Gym-style environment is used for training **one** robot policy with **PPO** (Stable-Baselines3).

**State / observation (single-agent):**
A compact kinematic observation encoding the robot–ball relationship and ball motion (position, distance, relative angle, velocity components).

**Action space (single-agent):**
A discrete set of sidestep actions (left/right at different speeds) plus a no-op/stand action.

**Reward design (single-agent):**
Dense shaping to reward:
- decreasing robot–ball distance,
- better alignment to the ball,
- and penalizing wasted steps,
with terminal conditions for successful interception vs timeout/failure.

**Training details (single-agent):**
Trained with PPO for a fixed training budget (e.g., hundreds of thousands of timesteps), using standard PPO components (rollout length, batch size, clipping, GAE, etc.) and periodic evaluation to select best checkpoints.

## Multi-agent approach (Ray RLlib PPO, shared policy)
A RLlib `MultiAgentEnv`-style wrapper trains **two** agents simultaneously using a **shared policy** (both robots learn the same policy weights) so that coordination emerges from observations + reward shaping.

**State / observation (multi-agent):**
Per-agent observations focus on each robot’s relative ball geometry (e.g., ball position and bearing in that agent’s frame).

**Action space (multi-agent):**
A small discrete set: **sidestep left**, **sidestep right**, **stand**.

**Reward design (multi-agent):**
A cooperative shaping scheme that:
- encourages the **closest** robot to engage/intercept,
- encourages the **farthest** robot to reposition to support,
- and discourages counterproductive motion,
so the pair learns complementary roles while sharing a single policy.

**Synchronized stepping (important implementation detail):**
The simulator is stepped only once both agents have submitted actions, and observations are collected in a single receive pass—this avoids sync deadlocks and keeps both agents aligned in time.

**Training details (multi-agent):**
RLlib PPO configuration is used (batching, clipping, entropy, GAE/λ, etc.) with a shared-policy mapping for both agents.

## Simulation configuration (SimSpark)
Training is performed with simulation settings aimed at stable and efficient RL:
- synchronized stepping,
- controlled noise configuration,
- non-realtime execution for speed,
- and monitoring/visualization support.

## Evaluation methodology (what to look at)
Models are evaluated using multiple stability/performance indicators such as:
- mean episode reward and its variance,
- mean episode length and its variance,
- and success rate of interception episodes (where applicable),
with “best” models selected via periodic evaluation/checkpointing.

## Sim2Real considerations
The project documents common gaps between simulation and real robots and mitigations, including:
- limited/partial ball visibility vs “cheat” state in sim,
- action jitter and oscillations (hysteresis / acting every N steps),
- compute and thermal limits (lightweight inference, possible ONNX export),
- and RoboCup SPL communication constraints (bandwidth/latency).


https://github.com/user-attachments/assets/2a55c67a-e564-49a7-a77c-9b0fd2a62759


## Acknowledgements
This repository is based on an academic project/report implementing RL interception behaviors in RoboCup SPL simulation and documenting both single-agent and multi-agent training pipelines.

