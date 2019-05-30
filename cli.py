import sys
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
import pickle
from task import Task
from agents.ddpg import Agent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#######################
# Settings

batch_name = "20190530_2245_"

plt.interactive(False)


#######################
# Helper functions

# 3d print by https://github.com/mohamedabdallah1996
def quadcopter_3d_plot(results, vars=['x', 'y', 'z'], title='', save_to=None):
    x = results[vars[0]]
    y = results[vars[1]]
    z = results[vars[2]]
    c = results['time']

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = plt.axes(projection='3d')
    cax = ax.scatter(x, y, z, c=c, cmap='YlGn')
    ax.set(xlabel=vars[0], ylabel=vars[1], zlabel=vars[2], title=title)
    fig.colorbar(cax, label='Time step (s)', pad=0.1, aspect=40)

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)


def quadcopter_3d_plot2(results, vars=['x', 'y', 'z'], title='', save_to=None):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = plt.axes(projection='3d')
    for i in range(len(results)):
        x = results[i][vars[0]]
        y = results[i][vars[1]]
        z = results[i][vars[2]]
        c = results[i]['time']
        ax.plot(x, y, z)

    ax.set(xlabel=vars[0], ylabel=vars[1], zlabel=vars[2], title=title)

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)


def make_df_rewards(all_results):
    rewards_summed = {}
    for key, value in all_results.items():
        rewards_summed[key] = [sum(value['reward'])]
    df = pd.DataFrame(rewards_summed, index=['Reward']).transpose()
    return df


def plot_rewards(all_results, save_to=None):
    df = make_df_rewards(all_results)
    fig = plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(df['Reward'])
    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)


def get_avg_score(results):
    score_per_episode = []
    for res in results:
        episodes = len(res)
        df = make_df_rewards(res)
        score_per_episode.append(df['Reward'].sum() / episodes)
    score_per_episode = round(float(np.mean(score_per_episode)), ndigits=0)
    return score_per_episode


########################
# Main function
def do_learning(
    batch="",
    num_episodes=800,
    runtime=5,
    # we want the quad copter to go high up!
    target_pos=np.array([0., 0., 35.]),
    init_pos=np.array([0., 0., 10., 0., 0., 0.]),
):

    task = Task(
        target_pos=target_pos,
        init_pose=init_pos,
        runtime=runtime
    )
    agent = Agent(
        task=task,
        exploration_mu=0, exploration_theta=0.15, exploration_sigma=0.2, tau=0.01
    )

    labels = [
        'time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity', 'y_velocity', 'z_velocity', 'phi_velocity',
        'theta_velocity', 'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward']

    all_results = {}

    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        results = {x: [] for x in labels}
        while True:
            # agent chooses action
            action = agent.act(state)
            # Trying to force no action does make quad copter just drop to the floor
            # action = [0,0,0,0]

            # Trying to force action full throttle upwards
            #action = [9000, 9000, 9000, 9000]

            # Feed action to environment and get next state
            next_state, reward, done = task.step(action)

            # Make agent learn from step
            agent.step(action, reward, next_state, done)

            # IMPORTANT - update state for next iteration
            state = next_state

            # save results
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action) + [reward]
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])

            if done:
                if i_episode % 100 == 0:
                    print("Done:  " + str(i_episode))
                    plot_rewards(all_results, save_to="plots\\"+batch+"_plot.png")

                    # filename = "plots\\"+batch+"_agent.pkl"
                    # outfile = open(filename, 'w')
                    # pickle.dump(agent, outfile)
                    # outfile.close()

                    filename = "plots\\" + batch + "_all_results.pkl"
                    outfile = open(filename, 'wb')
                    pickle.dump(all_results, outfile)
                    outfile.close()

                    # df_rew = make_df_rewards(all_results)
                    # quadcopter_3d_plot(
                    #     all_results[df_rew['Reward'].idxmax()],
                    #     save_to="plots\\"+batch+"_best_run.png")
                break
        all_results[i_episode] = results
        sys.stdout.flush()

    return all_results

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(do_learning)(batch=batch_name+"{:04d}".format(i)) for i in range(8))

print("avg score:    " + str(get_avg_score(results)))


# all_results = do_learning(
#     batch="batch_",
#     num_episodes=1000)
#
# df_rewards = make_df_rewards(all_results)
# df_rewards['Reward'].idxmax()
quadcopter_3d_plot2(results=[results[3][x] for x in range(1, 5)])
#
#
# for i in range(len(results)):
#     all_results = results[i]
#     batch = "batch_"+str(i)
#     plot_rewards(all_results, save_to="plots\\"+batch+"_plot.png")
#     df_rew = make_df_rewards(all_results)
#     quadcopter_3d_plot(
#        all_results[df_rew['Reward'].idxmax()],
#        save_to="plots\\"+batch+"_best_run.png")
#
