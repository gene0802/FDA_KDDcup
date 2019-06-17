from sys import exit, exc_info, argv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from netsapi.challenge import *

def plot(xs,ys,zs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xs, ys, zs, c='r', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

action_rewards = np.empty(0)
ITNS = np.empty(0)
IRS = np.empty(0)

class CustomAgent:
    # action_rewards = np.empty(100)
    # ITNS = np.empty(100)
    # IRS = np.empty(100)

    

    def __init__(self, environment):
        self.environment = environment

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        episode_avg_reward = []

        global action_rewards
        global ITNS
        global IRS
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in range(20):
                self.environment.reset()
                policy = {}
                rewards = np.empty(5)
                
                for j in range(5): #episode length
                    itns_num = random.random()
                    irs_num = random.random()

                    policy[str(j+1)]=[itns_num, irs_num]
                    state, reward, done, brace = self.environment.evaluateAction(policy[str(j+1)])
                    
                    action_rewards = np.append(action_rewards, reward)
                    ITNS = np.append(ITNS, itns_num) 
                    IRS = np.append(IRS, irs_num)

                    ##print(policy[str(j+1)], act_reward)
                episode_avg_reward.append(np.mean(rewards))
                candidates.append(policy)
                
            ##rewards = self.environment.evaluatePolicy(candidates)
            best_policy = candidates[np.argmax(episode_avg_reward)]
            best_reward = episode_avg_reward[np.argmax(episode_avg_reward)]
        except (KeyboardInterrupt, SystemExit):
            print(exc_info())
            
        return best_policy, best_reward

EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, "example.csv")

plot(ITNS, IRS, action_rewards)

