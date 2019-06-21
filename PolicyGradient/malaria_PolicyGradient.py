from sys import exit, exc_info, argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from netsapi.challenge import *
from GP_brain import PolicyGradient

Experiment_num = 0
action_buckets = (10,10)
episode_num = 20
state_num = 5
mode = "Policy Gradient"

RL = PolicyGradient(
    n_actions=2,
    n_features=1,
    learning_rate=0.02,
    reward_decay=0.99,
)
class CustomAgent:
    
    outputData = pd.DataFrame()
    def __init__(self, environment):
        self.environment = environment

    def append_Data(self,policy,rewards):
        s= pd.Series({'action1':policy['1'],'reward1':rewards[0], 
                              'action2':policy['2'],'reward2':rewards[1],
                              'action3':policy['3'],'reward3':rewards[2],
                              'action4':policy['4'],'reward4':rewards[3],
                              'action5':policy['5'],'reward5':rewards[4],
                              'total_reward':np.sum(rewards)})
        self.outputData = self.outputData.append(s,ignore_index=True)

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

        print("############################################################################################")
        print ("Experiment:\t"+str(Experiment_num))
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in range(episode_num):
                print("******************************************************************************")
                self.environment.reset()
                policy = {}
                states = {}
                rewards = np.empty(0)
                state = 0

                for j in range(5): #episode length
                    print("===========================")
                    print("j:\t"+str(j)) 

                    ## 抉擇action
                    action = RL.choose_action(np.asarray([state]))
                    print("action:\t"+str(action))

                    ## 獲取 reward , next state

                    next_state, reward, done, brace = self.environment.evaluateAction(action.tolist())
                    
                    print("reward:\t"+str(reward))

                    ##存transition table
                    RL.store_transition(next_state,[1,0], reward)
                    print(np.asarray([action[0],action[1]]))
            
                    ##
                    policy[str(j+1)] = action.tolist()
                    rewards = np.append(rewards,reward)
                    state = next_state

                episode_rewards.append(np.sum(rewards))
                episode_policies.append(policy)
                self.append_Data(policy,rewards)
                vt = RL.learn()
                if i == 0:
                    plt.plot(vt)    # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
           
            self.outputData.to_csv('./Result'+'/result_'+str(Experiment_num)+'.csv',float_format='%.2f')
            ##print (q_table) 
            ##rewards = self.environment.evaluatePolicy(candidates)
            best_policy = episode_policies[np.argmax(episode_rewards)]
            best_reward = episode_rewards[np.argmax(episode_rewards)]
            print("best_policy:"+str(best_policy))
            print("best_reward"+str(best_reward))
            



        except (KeyboardInterrupt, SystemExit):
            print(exc_info())
            
        return best_policy, best_reward
outputfile = './Submission'+'/submission_' + str(episode_num)+ "_" + str(action_buckets)+ ".csv"
EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, outputfile)




