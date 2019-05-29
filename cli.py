import sys
import numpy as np
import pandas as pd
from task import Task
from agents.ddpg import Agent

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#######################
# 3d print by https://github.com/mohamedabdallah1996
def quadcopter_3d_plot(results, vars=['x', 'y', 'z'], title=''):
    x = results[vars[0]]
    y = results[vars[1]]
    z = results[vars[2]]
    c = results['time']

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = plt.axes(projection='3d')
    cax = ax.scatter(x, y, z, c=c, cmap='YlGn')
    ax.set(xlabel=vars[0], ylabel=vars[1], zlabel=vars[2], title=title)
    fig.colorbar(cax, label='Time step (s)', pad=0.1, aspect=40)
    plt.show();



########################
num_episodes = 50

# we want the quad copter to go high up!
target_pos = np.array([0., 0., 100.])

init_pos = np.array([0., 0., 10., 0., 0., 0.])
task = Task(
    target_pos=target_pos,
    init_pose=init_pos
)
agent = Agent(
    task=task,
    exploration_mu=0, exploration_theta=0.15, exploration_sigma=0.2, tau=0.01
)

labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']

all_results = {}

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    results = {x: [] for x in labels}
    while True:
        # agent chooses action
        action = agent.act(state)
        # Trying to force no action does make quad copter just drop to the floor
        # action = [0,0,0,0]

        # Feed action to environment and get next state
        next_state, reward, done = task.step(action)

        # Make agent learn from step
        agent.step(action, reward, next_state, done)

        # IMPORTANT - update state for next iteration
        state = next_state

        # save results
        to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])

        if done:
            if i_episode % 50 == 0:
                print("Done:  " + str(i_episode))
            break
    all_results[i_episode] = results
    sys.stdout.flush()



quadcopter_3d_plot(all_results[len(all_results)], vars=['x', 'y', 'z'], title='')