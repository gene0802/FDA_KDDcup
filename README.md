# Execute
    python3 Qlearning/malaria_Qlearning.py
# Flow process
    1. choose action
    2. evaluate action
    3. update  Qtable
# Qlearning
    lr: learning rate
    gamma : discount factor
    epsilon : exploration rate
## Qtable 
     q_table = np.zeros((state_num,)+action_buckets)
        state_num : 5
        action_bucket : (10,10)
## update Qtable
     q_table[(state,) + action_num] += lr * (reward + gamma[j] * q_next_max - q_table[(state,)+ action_num])

    
