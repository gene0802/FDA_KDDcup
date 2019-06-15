from sys import exit, exc_info, argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
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


All_rewards = np.empty(0)
All_ITNS = np.empty(0)
All_IRS = np.empty(0)
action_buckets = (10,10)
episode_num = 20
state_num = 5
class CustomAgent:

    def __init__(self, environment):
        self.environment = environment

    def get_action(self,action, action_buckets):
        for i, s in enumerate(action): # 每個 feature 有不同的分配
            l, u = 0.0, 1.0 # 每個 feature 值的範圍上下限
            if s <= l: # 低於下限，分配為 0
                action[i] = 0
            elif s >= u: # 高於上限，分配為最大值
                action[i] = 10
            else: # 範圍內，依比例分配             
                action[i] = int(((s - l) / (u - l)) * action_buckets[i])
        return tuple(action)

    def choose_action(self, state, q_table, epsilon,action_buckets):
        if np.random.random_sample() < epsilon: # 有 ε 的機率會選擇隨機 action
            print("random action")
            return self.get_action([random.random(),random.random()],action_buckets)
        else: # 其他時間根據現有 policy 選擇 action，也就是在 Q table 裡目前 state 中，選擇擁有最大 Q value 的 action
           
            if np.amax(q_table[state])<= 20.0 :
                print("random action")
                return self.get_action([random.random(),random.random()],action_buckets)
            else :
                print("table action")
                result = np.where(q_table[state] == np.amax(q_table[state]))
                return (result[0][0], result[1][0]) 

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        episode_policies = []
        episode_rewards = []

        global All_rewards
        global All_ITNS
        global All_IRS

        global action_buckets 
        global episode_num 
        global state_num 

        get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/(episode_num/10.0))))  # epsilon-greedy; 隨時間遞減
        get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/(episode_num/10.0)))) # learning rate; 隨時間遞減 
        #gamma = 0.99 # reward discount factor
        gamma =[0.8,0.6,0.4,0.2,0.0]
        q_table = np.zeros((state_num,)+action_buckets)




        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in range(episode_num):
                print("******************************************************************************")
                self.environment.reset()
                policy = {}
                rewards = np.empty(0)
                state = 0
                epsilon = get_epsilon(i)
                lr = get_lr(i)
                print ("epsilon:\t")    
                print(epsilon)
                print("lr:\t")  
                print(lr)
                

                for j in range(5): #episode length
                    print("===========================")
                    print("j:\t"+str(j)) 


                    ## 抉擇action
                    action_num = self.choose_action(state, q_table, epsilon,action_buckets)
                    action = [action_num[0]/action_buckets[0], action_num[1]/action_buckets[1]]
                   

                    ## 獲取 reward , next state
                    next_state, reward, done, brace = self.environment.evaluateAction(action)
                    print("action:\t"+str(action)) 
                    print("reward:\t"+str(reward)) 


                    ##更新q_table
                    
                    if next_state-1 <5:
                        q_next_max = np.amax(q_table[next_state-1])
                    else:   
                        q_next_max = 0.0

                    q_table[(state,) + action_num] += lr * (reward + gamma[j] * q_next_max - q_table[(state,)+ action_num]) 
                    
                    ##
                    policy[str(j+1)] = action
                    rewards = np.append(rewards,reward)
                    state = next_state-1
                    
                    #All_rewards = np.append(All_rewards, reward)
                    #All_ITNS = np.append(All_ITNS, ITNS) 
                    #All_IRS = np.append(All_IRS, IRS)

                episode_rewards.append(np.sum(rewards))
                episode_policies.append(policy)
            print ("q_table")
            print (q_table)  
            ##rewards = self.environment.evaluatePolicy(candidates)
            best_policy = episode_policies[np.argmax(episode_rewards)]
            best_reward = episode_rewards[np.argmax(episode_rewards)]
        except (KeyboardInterrupt, SystemExit):
            print(exc_info())
            
        return best_policy, best_reward
outputfile = "example_" + str(episode_num)+ "_" + str(action_buckets)+ ".csv"
EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, outputfile)

##plot(ITNS, IRS, action_rewards)



