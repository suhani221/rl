# Function to choose M projects based on estimated Whittle indices
from newstate import new_state_transition
import numpy as np 
from chooseproject import choose_projects
from initilisation import initial_1
#import matplotlib.pyplot as plt



alpha = 0.3
def calculate_whittle_indices(QA, QP, HA, HP, K):

    indices = (QA - QP)
    return indices


def generate_random_array(N, M):
    # Create an array of zeros of length N
    array = np.zeros(N, dtype=int)
    
    # Randomly select M indices to set to 1
    indices_to_set = np.random.choice(N, M, replace=False)
    
    # Set the selected indices to 1
    array[indices_to_set] = 1
    
    return array

def choose_projects(whittle_indices):
    M = 3
    final_whittle = []
    for array in whittle_indices:
        print(array)
        finite_values = array[np.isfinite(array)]
        finite_sum = np.sum(finite_values)
        final_whittle.append(finite_sum)
    
    sorted_projects = np.argsort(final_whittle)
    active_projects = np.argsort(sorted_projects)[-M:]
    new_action = []
    for i in sorted_projects:
        if(i in active_projects):
            new_action.append(1)
        else:
            new_action.append(0)
    print(new_action)
    return final_whittle, sorted_projects, active_projects, new_action


def simulation_1(N, K, TPA, TPP, QA, QP, HA, HP, whittle_indices_initial, revsys, worsys, policy, current_state):
    M = 3
    whittle_indices_total = []
    reward = []
    for t in range(400): 
        
        print(f'Simulation Number : {t}')
        print(f'Simulation Number {t} | Current State : {current_state}')
        # if np.random.rand() < 0.2:  
        #     new_action = generate_random_array(N, M)
        # else:
        if(t==0):
            project_scores, sorted_projects, active_projects, new_action = choose_projects(whittle_indices_initial)
            print("New Action:", new_action)
            print(f'Simulation Number {t} | New Action : {new_action}')
            whittle_indices_total.append(project_scores)
            previous_state = current_state.copy()
            print(f'This is the first {previous_state}')
            new_state = new_state_transition(current_state, transition_prob_active = TPA, transition_prob_passive = TPP, N=5, K=5, action = new_action)
            print(f'This is the first {new_state}')
            print(f'Simulation Number {t} | New State : {new_state}')
        else:
            if np.random.rand() < 0.25:  
                new_action = generate_random_array(N, M)
                print(f'Simulation Number {t} | New Action : {new_action}')
                previous_state = new_state.copy()
                new_state = new_state_transition(new_state, transition_prob_active = TPA, transition_prob_passive = TPP, N=5, K=5, action = new_action)
                print(f'Simulation Number {t} | New State : {new_state}')
            else:
                whittle_indices = calculate_whittle_indices(QA, QP, HA, HP, K)
                project_scores, sorted_projects, active_projects, new_action = choose_projects(whittle_indices)
                print(f'Simulation Number {t} | New Action : {new_action}')
                whittle_indices_total.append(project_scores)
                previous_state = new_state.copy()
                new_state = new_state_transition(new_state, transition_prob_active = TPA, transition_prob_passive = TPP, N=5, K=5, action = new_action)
                print(f'Simulation Number {t} | New State : {new_state}')
        print('--------------------------------------------')
        print(f'check', previous_state)
        print(f'check', new_state)
        count = 0
        count = 0
        print('--------------------------------------------')
        print(f'check', previous_state)
        print(f'check', new_state)
        for arm, action in enumerate(new_action):
        
            count = count + 1
            QAAVG = QA[arm, :]
            HAAVG = HA[arm, :]
            QPAVG = QA[arm, :]
            HPAVG = HA[arm, :]
            if (action == 1):
                Prev_state = previous_state[arm]
                Next_state = new_state[arm]
                reward.append(revsys[Next_state])
                print(f'Simulation Number {t} | Prev : {Prev_state} - {count}')
                print(f'Simulation Number {t} | Next : {Next_state} - {count}')
                print(f'Simulation Number {t} | reward for the next state : {revsys[Prev_state]} - {count}')
                
                QA[arm, Prev_state] = QA[arm, Prev_state] + alpha * (revsys[Next_state] + QA[arm, Next_state] - (1 / (2 * N)) * (np.sum(QAAVG + QPAVG)) - QA[arm, Prev_state])
                HA[arm, Prev_state] = HA[arm, Prev_state] + alpha * (worsys[Next_state] + HA[arm, Next_state] - (1 / (2 * N)) * (np.sum(HAAVG + HPAVG)) - HA[arm, Prev_state])
            elif(action == 0):
                Prev_state = previous_state[arm]
                Next_state = new_state[arm]
                reward.append(revsys[Next_state])
                print(f'Simulation Number {t} | Prev : {Prev_state} - {count}')
                print(f'Simulation Number {t} | Next : {Next_state} - {count}')
                print(f'Simulation Number {t} | reward for the next state : {revsys[Prev_state]} - {count}')
                QP[arm, Prev_state] = QP[arm, Prev_state] + alpha * (revsys[Next_state] + QP[arm, Next_state] - (1 / (2 * N)) * (np.sum(QAAVG + QPAVG)) - QP[arm, Prev_state])
                HP[arm, Prev_state] = HP[arm, Prev_state] + alpha * (worsys[Next_state] + HP[arm, Next_state] - (1 / (2 * N)) * (np.sum(HAAVG + HPAVG)) - HP[arm, Prev_state])
            else:
                print("Nothing here")
        print(f'Simulation Number Finished: {t}')

    return QA, QP, HA, HP, whittle_indices_total, reward


def main():
    


    M = 3
    N = 5
    K = 5
    N_1, K_1, TPA_1, TPP_1, QA_1, QP_1, HA_1, HP_1, whittle_indices_initial_1, revsys_1, worsys_1, policy_1, current_state_1 = initial_1(N, K)
    QA_new_1, QP_new_1, HA_new_1, HP_new_1, whittle_index_1, reward= simulation_1(N_1, K_1, TPA_1, TPP_1, QA_1, QP_1, HA_1, HP_1, whittle_indices_initial_1, revsys_1, worsys_1, policy_1, current_state_1)
    whittle_index = np.array(whittle_index_1)
    plt.plot(whittle_index.transpose()[0], label='Arm 1')
    plt.plot(whittle_index.transpose()[1], label='Arm 2')
    plt.plot(whittle_index.transpose()[2], label='Arm 3')
    plt.plot(whittle_index.transpose()[3], label='Arm 4')
    plt.plot(whittle_index.transpose()[4], label='Arm 5')
    plt.xlim(0, 350)
    plt.ylim(-50,50)
    # Add labels for x and y axes
    plt.xlabel('Time - 100 Epochs')
    plt.ylabel('Whittle Index')


    # Add legend
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()