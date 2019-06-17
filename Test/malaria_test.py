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

Experiment_num = 0
action_buckets = (10,10)
episode_num = 20
state_num = 5
mode = "testing"

class CustomAgent:
    
    outputData = pd.DataFrame()
    def __init__(self, environment):
        self.environment = environment

    def get_action(self,action, action_buckets):
        for i, s in enumerate(action): # 每個 feature 有不同的分配
            l, u = 0.0, 1.0 # 每個 feature 值的範圍上下限
            if s <= l: # 低於下限，分配為 0
                action[i] = 0.0
            elif s >= u: # 高於上限，分配為最大值
                action[i] = 10.0
            else: # 範圍內，依比例分配             
                action[i] = int(((s - l) / (u - l)) * action_buckets[i])
        return tuple(action)

    def choose_action(self, state):
        if state%2 == 0:
            return [1.0,0.0]
        else :
            return [0.0,1.0]

    def append_Data(self,policy,rewards):
        s= pd.Series({'action1':policy['1'],'reward1':rewards[0], 
                              'action2':policy['2'],'reward2':rewards[1],
                              'action3':policy['3'],'reward3':rewards[2],
                              'action4':policy['4'],'reward4':rewards[3],
                              'action5':policy['5'],'reward5':rewards[4],
                              'total_reward':np.sum(rewards)})
        self.outputData = self.outputData.append(s,ignore_index=True)
        return 1
    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        episode_policies = []
        episode_rewards = []

        global Experiment_num
        global action_buckets 
        global episode_num 
        global state_num 
        Experiment_num  += 1
        get_epsilon = lambda i: max(0.01, min(1, math.log10((episode_num-i)/(episode_num/10.0))))  # epsilon-greedy; 隨時間遞減
        get_lr = lambda i: max(0.01, min(0.5, math.log10((episode_num-i)/(episode_num/10.0)))) # learning rate; 隨時間遞減 

        gamma =[0.0,0.0,0.0,0.0,0.0]
        q_table = np.zeros((state_num,)+action_buckets)
        print("############################################################################################")
        #print ("Experiment:\t"+str(Experiment_num))
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in range(episode_num):
                print("******************************************************************************")
                self.environment.reset()
                policy = {}
                states = {}
                rewards = np.empty(0)

                state = 0
                #epsilon = get_epsilon(i)
                #lr = get_lr(i)

                #print ("epsilon:\t"+str(epsilon))    
                #print("lr:\t"+str(lr))  
                
                for j in range(5): #episode length
                    print("===========================")
                    print("j:\t"+str(j)) 


                    ## 抉擇action
                    action = self.choose_action(state)
                    #action = [action_num[0]/action_buckets[0], action_num[1]/action_buckets[1]]

                    ## 獲取 reward , next state
                    next_state, reward, done, brace = self.environment.evaluateAction(action)
                    print("action:\t"+str(action)) 
                    print("reward:\t"+str(reward)) 

                    
                    ##
                    policy[str(j+1)] = action
                    rewards = np.append(rewards,reward)
                    state = next_state-1

                episode_rewards.append(np.sum(rewards))
                episode_policies.append(policy)
                self.append_Data(policy,rewards)
           
            self.outputData.to_csv('./Result/'+'Result_'+str(Experiment_num)+'.csv',float_format='%.2f')
            ##print (q_table) 
            ##rewards = self.environment.evaluatePolicy(candidates)
            best_policy = episode_policies[np.argmax(episode_rewards)]
            best_reward = episode_rewards[np.argmax(episode_rewards)]
        except (KeyboardInterrupt, SystemExit):
            print(exc_info())
            
        return best_policy, best_reward
outputfile = "./Submission/"+ mode + "/submission_" + str(episode_num)+ "_" + str(action_buckets)+ ".csv"
EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, outputfile)

##plot(ITNS, IRS, action_rewards)
