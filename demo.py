import numpy as np
# from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from utils import *
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import pybullet
import time
import model
import filter
import json
from Visualization import vis


def executeBase(robot, joints, sleep):
    # NOTE: Only for timer!
    path = model.Path_Real
    for bq in path:
        set_joint_positions(robot, joints, bq)
        wait_for_duration(sleep)

#         marker_pos_sense = bq
#         draw_sphere_marker(marker_pos_sense, 0.1, (1, 0, 0, 1))

# #       Filtered Trajecgtory: Blue
#         marker_pos_filter = bq
#         marker_pos_filter.append(1.4)
#         draw_sphere_marker(marker_pos_filter, 0.1, (0, 0, 1, 1))

#       Groundtruth: Green
        marker_pos_real = bq
        marker_pos_real[2] = 0
        draw_sphere_marker(marker_pos_real, 0.1, (0, 1, 0, 1))
    print('Finished')


def executeKalman(robot, joints, sleep):
    mu = np.array(model.Path_Real[0][:2])
    Sigma = np.zeros(mu.shape)
    Real_Path = []
    Sense_Path = []
    Filtered_Path = []
    for i in range(1, len(model.Path_Real)):
        u = np.array(np.array(
            model.Path_Action[i][:2]) - np.array(model.Path_Action[i-1][:2]))  # action
        # z = np.array(model.Path_Real[i][:2])
        # z[0] += np.random.normal(0, 1.)  # observation noise
        # z[1] += np.random.normal(0, 1.)  # observation noise
        z = np.array(model.SensorInput[i][:2])
        set_joint_positions(robot, joints, model.Path_Real[i])
        mu, Sigma = filter.KalmanFilter(mu, u, z, Sigma)
        wait_for_duration(sleep)

#       Sensed Trajecgtory: Red
        marker_pos_sense = z.tolist()
        marker_pos_sense.append(1.4)
        draw_sphere_marker(marker_pos_sense, 0.1, (1, 0, 0, 1))

#       Filtered Trajecgtory: Blue
        marker_pos_filter = mu.tolist()
        marker_pos_filter.append(1.4)
        draw_sphere_marker(marker_pos_filter, 0.1, (0, 0, 1, 1))

#       Groundtruth: Green
        marker_pos_real = model.Path_Real[i]
        marker_pos_real[2] = 1.4
        draw_sphere_marker(marker_pos_real, 0.1, (0, 1, 0, 1))

#       Store
        Sense_Path.append(z.tolist())
        Filtered_Path.append(mu.tolist())
        Real_Path.append(model.Path_Real[i][:2])
    with open("Data/KalmanSensePath.json",'w') as f:
        json.dump(Sense_Path, f, indent=2) 
    with open("Data/KalmanFilteredPath.json",'w') as f:
        json.dump(Filtered_Path, f, indent=2) 
    with open("Data/KalmanRealPath.json",'w') as f:
        json.dump(Real_Path, f, indent=2) 
    print('Finished')


def executeParticle(robot, joints, sleep):
    mu = np.array(model.Path_Real[0][:2])
    # Sigma = np.zeros(mu.shape)
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])

    # initialize particles
    M = 1000 # number of particles
    particles = np.zeros((M, 2))

    Real_Path = []
    Sense_Path = []
    Filtered_Path = []
    for particle_i in range(M):
        particle_sampled = np.random.multivariate_normal(mu, Sigma)
        while filter.checkCollision(particle_sampled):
            particle_sampled = np.random.multivariate_normal(mu, Sigma)
        particles[particle_i] = particle_sampled
    w = np.ones(M) / M

    # for particle_id, particle in enumerate(particles):
    #     marker_particle = list(particle)
    #     marker_particle.append(1.6)
    #     # print(marker_particle)
    #     # input()
    #     # marker_particle[2] = 1.4
    #     # np.random.seed(particle_id)
    #     draw_sphere_marker(marker_particle, 0.1, (0, particle_id/M, 1, 1))
    # input()

    for i in range(1, len(model.Path_Real)):
        u = np.array(np.array(
            model.Path_Action[i][:2]) - np.array(model.Path_Action[i-1][:2]))  # action
        # z = np.array(model.Path_Real[i][:2])
        # z[0] += np.random.normal(0, 1.)  # observation noise
        # z[1] += np.random.normal(0, 1.)  # observation noise
        z = np.array(model.SensorInput[i][:2])
        set_joint_positions(robot, joints, model.Path_Real[i])
        particles, w = filter.ParticleFilter(M, mu, u, z, particles, w)
        mu = (particles * w.reshape(-1, 1)).sum(axis=0)
        wait_for_duration(sleep)

        
#       Sensed Trajecgtory: Red
        marker_pos_sense = z.tolist()
        marker_pos_sense.append(1.4)
        draw_sphere_marker(marker_pos_sense, 0.1, (1, 0, 0, 1))

#       Filtered Trajecgtory: Blue
        marker_pos_filter = mu.tolist()
        marker_pos_filter.append(0.8)
        draw_sphere_marker(marker_pos_filter, 0.1, (0, 0, 1, 1))

#       Groundtruth: Green
        marker_pos_real = model.Path_Real[i]
        marker_pos_real[2] = 1.4
        draw_sphere_marker(marker_pos_real, 0.1, (0, 1, 0, 1))

#       Store
        Sense_Path.append(z.tolist())
        Filtered_Path.append(mu.tolist())
        Real_Path.append(model.Path_Real[i][:2])

        # for particle_id, particle in enumerate(particles):
        #     marker_particle = list(particle)
        #     marker_particle.append(1.6)
        #     # print(marker_particle)
        #     # input()
        #     # marker_pos_real[2] = 1.4
        #     draw_sphere_marker(marker_particle, 0.1, (0, particle_id/M, 1, 1))
        # input()

    with open("Data/ParticleSensePath"+str(M)+".json",'w') as f:
        json.dump(Sense_Path, f, indent=2) 
    with open("Data/ParticleFilteredPath"+str(M)+".json",'w') as f:
        json.dump(Filtered_Path, f, indent=2) 
    with open("Data/ParticleRealPath"+str(M)+".json",'w') as f:
        json.dump(Real_Path, f, indent=2) 
    print('Finished')

def main(screenshot=False):
    np.random.seed(42)

    input("Press Enter to view data visualization of our result")
    vis()

    input("Press Enter to view localization demo of Kalman Filter")
    # 1
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2maze.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name)
                   for name in PR2_GROUPS['base']]


    pybullet.resetDebugVisualizerCamera( cameraDistance=10, cameraYaw=0, cameraPitch=270.1, cameraTargetPosition=[3.5,-1,0])
    t0 = time.time()
    executeKalman(robots['pr2'], base_joints,
                  sleep=0.00)
    wait_if_gui()
    disconnect()


    input("Press Enter to view localization demo of Particle Filter with 1000 particles")
    # 2
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2maze.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name)
                   for name in PR2_GROUPS['base']]

    pybullet.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=270.1, cameraTargetPosition=[3.5,-1,0])
    t0 = time.time()
    # executeKalman(robots['pr2'], base_joints,
                #   sleep=0.00)
    executeParticle(robots['pr2'], base_joints,
                    sleep=0.01)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
