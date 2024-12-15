#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import pybullet as p
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
from gym_pybullet_drones.envs import ObS12Stage1
from gym_pybullet_drones.envs import ObS12Stage2
from gym_pybullet_drones.envs import ObS12Stage3
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool
import matplotlib.pyplot as plt


def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))

def letter_to_points(letter, scale=1.0):
    """
    Define key points for each letter in the XZ plane.
    The drone first hovers to the top-left corner (point 1) before tracing the letter.
    """
    if letter == 'A':
        return [
            [0.5, 0, 2 * scale],  # Hover to top-center (Point 1)
            [0, 0, 0.5],            # Move to bottom-left
            [0.5, 0, 2 * scale],  # Move to top-center
            [1, 0, 0.5],            # Move to bottom-right
            [0.25, 0, 1.25 * scale],  # Crossbar start
            [0.75, 0, 1.25 * scale]   # Crossbar end
        ]
    elif letter == 'B':
        return [
            [0, 0, 1 * scale],    # Hover to top-left corner (Point 1)
            [0, 0, 0],            # Bottom-left
            [0, 0, 1 * scale],    # Top-left
            [0.6, 0, 1 * scale],  # Top-right of upper loop
            [1, 0, 0.75 * scale], # Outer curve of upper loop
            [0.6, 0, 0.5 * scale],# Middle of letter
            [0, 0, 0.5 * scale],  # Center of the loops
            [0.6, 0, 0.5 * scale],# Start lower loop
            [1, 0, 0.25 * scale], # Bottom-right curve
            [0.6, 0, 0],          # Back to bottom-left
            [0, 0, 0]             # Complete the shape
        ]
    elif letter == 'C':
        return [
            [1, 0, 1 * scale],    # Hover to top-right (Point 1)
            [0.5, 0, 1 * scale],  # Move to upper-middle
            [0, 0, 0.5 * scale],  # Move to center-left
            [0.5, 0, 0],          # Move to bottom-middle
            [1, 0, 0]             # Move to bottom-right
        ]
    elif letter == 'D':
        return [
            [0, 0, 1 * scale],    # Hover to top-left corner (Point 1)
            [0, 0, 0],            # Bottom-left
            [0, 0, 1 * scale],    # Top-left
            [0.5, 0, 1 * scale],  # Top-right curve
            [1, 0, 0.5 * scale],  # Center of curve
            [0.5, 0, 0],          # Bottom-right curve
            [0, 0, 0]             # Back to bottom-left
        ]
    else:
        raise ValueError(f"Letter '{letter}' is not defined!")
    
def plot_target_trajectory(x, y, z, ax=None):
    font = {'family': 'serif', 'weight': 'normal', 'size': 15}
    plt.rc('font', **font)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    ax.plot(x, y, z, color='blue', linewidth=2, label='Target Trajectory')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-35, roll=0)
    ax.legend()
    plt.show()


def get_policy(policy_path, model):
    if os.path.isfile(policy_path + '/best_model.zip'):
    # if os.path.isfile(policy_path + '/' + model):
        return PPO.load(policy_path + '/best_model.zip')
        # return PPO.load(policy_path + '/' + model)

    raise Exception("[ERROR]: no model under the specified path", policy_path)


def spiral_trajectory(number_of_points: int = 50, radius: int = 2, angle_range: float = 30.0) -> tuple:
    angles = np.linspace(0, 4 * np.pi, number_of_points)
    x_coordinates = radius * np.cos(angles)
    y_coordinates = radius * np.sin(angles)
    z_coordinates = np.linspace(0, 1, number_of_points)

    yaw_angles = np.arctan2(y_coordinates, x_coordinates)

    angle_range_rad = np.radians(angle_range)

    oscillation = angle_range_rad * np.sin(np.linspace(0, 4 * np.pi, number_of_points))

    yaw_angles += oscillation

    yaw_angles = np.clip(yaw_angles, -angle_range_rad, angle_range_rad)

    return x_coordinates, y_coordinates, z_coordinates, yaw_angles

def lemniscata_trajectory(number_of_points: int = 50, a: float = 2) -> tuple:
    t = np.linspace(0, 2.125 * np.pi, number_of_points)
    x_coordinates = a * np.sin(t) / (1 + np.cos(t)**2)
    y_coordinates = a * np.sin(t) * np.cos(t) / (1 + np.cos(t)**2)
    z_coordinates = np.zeros(number_of_points)

    yaw_angles = np.arctan2(-y_coordinates, -x_coordinates)

    return x_coordinates, y_coordinates, z_coordinates, yaw_angles


def smooth_trajectory(points, num_points=100):
    points = np.array(points)
    tck, u = splprep([points[:,0], points[:,1], points[:,2]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    smooth_points = np.vstack((x_fine, y_fine, z_fine)).T

    tangents = np.diff(smooth_points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    tangents = np.vstack((tangents, tangents[-1]))

    roll_pitch_yaw = []

    for i in range(len(smooth_points)):
        t = tangents[i]
        y_axis = t
        z_axis = np.array([0, 0, 1])
        if np.allclose(y_axis, z_axis):
            z_axis = np.array([0, 1, 0])
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        r = R.from_matrix(rotation_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        roll_pitch_yaw.append((roll, pitch, yaw))

    x_tuple = tuple(smooth_points[:, 0])
    y_tuple = tuple(smooth_points[:, 1])
    z_tuple = tuple(smooth_points[:, 2])
    yaw_tuple = tuple([yaw for _, _, yaw in roll_pitch_yaw])

    return x_tuple, y_tuple, z_tuple, yaw_tuple


def run_simulation(
        test_env,
        policy_path,
        model='best_model.zip',
        gui=True,
        record_video=True,
        reset=False,
        save=True,
        plot=False,
        debug=False,
        comment=""
):
    policy = get_policy(policy_path, model)
    # drone_positions = []
    # target_positions = []

    test_env = test_env(
                 initial_xyzs=np.array([[0.5, 0, 0]]),
                 initial_rpys=np.array([[0, 0, 0]]),
        gui=gui,
                        obs=ObservationType('kin'),
                        act=ActionType('rpm'),
                        record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=policy_path,
        colab=False
    )

    obs, info = test_env.reset(options={})
    simulation_length = (test_env.EPISODE_LEN_SEC + 30) * test_env.CTRL_FREQ

    start = time.time()
    # Parameters
    camera_distance = 4  # Distance from the target
    camera_yaw = 0      # Yaw angle (degrees)
    camera_pitch = -25   # Pitch angle (degrees)
    camera_target = [0.5, 0, 1]  # The point the camera looks at (x, y, z)

    # Reset the camera
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target,
    )

    # hover time
    hover_time= int(0.1 * simulation_length)
    hover_position = [0.5, 0, 2]
    # start_pos= [0.5, 0, 0]


    all_x_target = []
    all_y_target = []
    all_z_target = []
    all_yaw_target = []
    # for word in words:

    #     points = letter_to_points(word, scale=1)
    #     xt_target, yt_target, zt_target, yawt_target = smooth_trajectory([start_pos, points[0]], num_points=transition_length)
    #     x_target, y_target, z_target, yaw_target = smooth_trajectory(points, num_points=char_length)
    #     all_x_target.extend(xt_target)
    #     all_y_target.extend(yt_target)
    #     all_z_target.extend(zt_target)
    #     all_yaw_target.extend(yawt_target)
    #     all_x_target.extend(x_target)
    #     all_y_target.extend(y_target)
    #     all_z_target.extend(z_target)
    #     all_yaw_target.extend(yaw_target)
    #     start_pos= points[-1]

    # x_target, y_target, z_target, yaw_target = spiral_trajectory(simulation_length, 2)
    # x_target, y_target, z_target, yaw_target = spiral_trajectory(simulation_length, 2)
    # x_target, y_target, z_target, yaw_target = lemniscata_trajectory(simulation_length, 2)
    # points = [

    #     [2,0,1],
    #     [3, 0, 3],
    #     [4, 0, 1],
    #     [3.5, 0, 2],
    #     [2.5, 0, 2]
    # ]
    points = letter_to_points('A', scale=1)
    # xt_target, yt_target, zt_target, yawt_target = smooth_trajectory([[0.5,0,0], points[0]], num_points=500,k=3)
    # hover=[]
    # hover.append([0.5, 0, 0])
    # hover.append(points[0]/2)
    # hover.append(points[0])
    x_target, y_target, z_target, yaw_target = smooth_trajectory(points, num_points=simulation_length)
    # all_x_target.append(xt_target)
    # all_y_target.append(yt_target)
    # all_z_target.append(zt_target)
    # all_yaw_target.append(yawt_target)
    # all_x_target.append(x_target)
    # all_y_target.append(y_target)
    # all_z_target.append(z_target)
    # all_yaw_target.append(yaw_target)


    for i in range(simulation_length):
        # WAY-POINT TRACKING
        # if i < simulation_length / 5:
        #     x_target = -1
        #     y_target = 1
        #     z_target = 0
        #     yaw_target = -0.52
        # elif simulation_length / 5 < i < 2 * simulation_length / 5:
        #     x_target = -2
        #     y_target = 0
        #     z_target = 0.5
        #     yaw_target = 0
        # elif 2 * simulation_length / 5 < i < 3 * simulation_length / 5:
        #     x_target = -2
        #     y_target = -2
        #     z_target = 1.5
        #     yaw_target = 0.35
        # elif 3 * simulation_length / 5 < i < 4 * simulation_length / 5:
        #     x_target = -1
        #     y_target = -3
        #     z_target = 0
        #     yaw_target = 0.7
        # else:
        #     x_target = -2.8
        #     y_target = -3.8
        #     z_target = 1
        #     yaw_target = 0
        #
        # obs[0][0] -= x_target
        # obs[0][1] -= y_target
        # obs[0][2] -= z_target
        # obs[0][5] -= 1

        # if i < hover_time:
        #     obs[0][0] -= hover_position[0]
        #     obs[0][1] -= hover_position[1]
        #     obs[0][2] -= hover_position[2]
        # else:
        #     idx=i - hover_time
        #     obs[0][0] -= x_target[idx]
        #     obs[0][1] -= y_target[idx]
        #     obs[0][2] -= z_target[idx]
            # obs[0][5] -= yaw_target
        # TRAJECTORY TRACKING
        obs[0][0] -= x_target[i]
        obs[0][1] -= y_target[i]
        obs[0][2] -= z_target[i]
        # obs[0][5] -= yaw_target[i]

                # TRAJECTORY TRACKING
        # obs[0][0] -= all_x_target[i]
        # obs[0][1] -= all_y_target[i]
        # obs[0][2] -= all_z_target[i]

        action, _states = policy.predict(obs,
                                         deterministic=True
                                         )

        obs, reward, terminated, truncated, info = test_env.step(action)
        actions = test_env._getDroneStateVector(0)[16:20]
        actions2 = actions.squeeze()
        obs2 = obs.squeeze()

        if debug:
            print(f"""
            #################################################################
            Observation Space:
            Position: {obs[0][0:3]}
            Orientation: {in_degrees(obs[0][3:6])}
            Linear Velocity: {obs[0][6:9]}
            Angular Velocity: {obs[0][9:12]}
            -----------------------------------------------------------------
            Action Space: type {type(action)} value {action}
            Terminated: {terminated}
            Truncated: {truncated}
            -----------------------------------------------------------------
            Policy Architecture: {policy.policy}
            #################################################################
            """)

        logger.log(
            drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=np.hstack([obs2[0:3],
                             np.zeros(4),
                             obs2[3:12],
                             actions2
                             ]),
            reward=reward,
            control=np.zeros(12)
        )

        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if reset and terminated:
            obs, info = test_env.reset(seed=42, options={})

    test_env.close()

    if plot:
        print("Plotting")
        # logger.plot_position_and_orientation()
        # logger.plot()
        # logger.plot_rpms()
        ax= logger.plot_trajectory()
        plot_target_trajectory(x_target, y_target, z_target,ax)

    if save:
        print("Saving")
        logger.save_as_csv(comment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a simulation given a trained policy")
    parser.add_argument(
        '--policy_path',
        default='/Users/shari/Documents/sem 3/rl_13th/gym-pybullet-drones/gym_pybullet_drones/experiments/continuous_learning',
        help='The path to a zip file containing the trained policy'
    )
    parser.add_argument(
        '--model',
        default='best_model.zip',
        help='The zip file containing the trained policy'
    )
    parser.add_argument(
        '--test_env',
        default=ObS12Stage1,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--gui',
        default=True,
        type=str2bool,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--record_video',
        default=False,
        type=str2bool,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--reset',
        default=False,
        type=str2bool,
        help="If you want to reset the environment, every time that the drone achieve the target position"
    )
    parser.add_argument(
        '--save',
        default=False,
        type=str2bool,
        help='Allow to save the trained data using csv and npy files'
    )
    parser.add_argument(
        '--comment',
        default="",
        type=str,
        help="A comment to describe de simulation saved data"
    )
    parser.add_argument(
        '--plot',
        default=True,
        type=str2bool,
        help="If are shown demo plots"
    )
    parser.add_argument(
        '--debug',
        default=False,
        type=str2bool,
        help="Prints debug information"
    )
    # parser.add_argument(
    #     '--apply_filter',
    #     default=False,
    #     type=str2bool,
    #     help="Applies a low pass to the actions coming from the policy"
    # )

    run_simulation(**vars(parser.parse_args()))
