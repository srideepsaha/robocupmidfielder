from agent.Base_Agent import Base_Agent as Agent
import time
import numpy as np

# Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name
player = Agent("localhost", 3100, 3200, 7, 1, "TeamName")

w = player.world

# Move the ball to position (1.0, 0.0, 0.0) with zero velocity
player.scom.unofficial_move_ball((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

# Allow time for the server to process the monitor command
for _ in range(5):
    player.scom.unofficial_beam((-3, 0.1, 0.5), 0)
    player.behavior.execute("Zero")
    player.scom.commit_and_send(w.robot.get_command())
    player.scom.receive()

# Start main loop

kick = False
while True:
    w = player.world
    robot_pos = w.robot.loc_head_position[:2]
    rel_ball_pos = w.ball_rel_torso_cart_pos[:2]
    print("Relative Ball", rel_ball_pos)
    init = [0.0, 0.0]
    ball_heading = np.arctan2(rel_ball_pos[1], rel_ball_pos[0])
    torso_orientation = np.deg2rad(w.robot.imu_torso_orientation)
    heading_diff = ball_heading - torso_orientation
    heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
    print("Angle is", heading_diff)
    print("new angle is", ball_heading)
    distance = np.linalg.norm(rel_ball_pos - init)
    #print("distance is", distance)
    #player.behavior.execute("Walk", w.ball_abs_pos[:2], True, None, True, None)
    #player.behavior.execute("Zero")
    if kick == False:
        player.behavior.execute_sub_behavior("Kick_Motion", True)
        kick = True
    else:
        player.behavior.execute_sub_behavior("Kick_Motion", False)
    player.scom.commit_and_send(w.robot.get_command())
    player.scom.receive()
