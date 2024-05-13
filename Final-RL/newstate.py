import numpy as np
import numpy.random as random

def new_state_transition(current_state, transition_prob_active, transition_prob_passive, N, K, action):
    
    for arm, state in enumerate(current_state):
        if(action[arm]==1):
            transition_row = transition_prob_active[state]

            two_probsa = []
            for possible_state in transition_row:
                if(possible_state!=0):
                    two_probsa.append(possible_state)
            number = random.random()
            if number < 0.85:
                choice = min(two_probsa)
                for i, prob in enumerate(transition_row):
                    if(prob == choice):
                        index = i
                        continue
            else:
                choice = max(two_probsa)
                for i, prob in enumerate(transition_row):
                    if(prob == choice):
                        index = i
                        continue            
            current_state[arm] = index   

        elif(action[arm]==0):
            transition_row = transition_prob_passive[state]
            two_probsp = []
            for possible_state in transition_row:
                if(possible_state!=0):
                    two_probsp.append(possible_state)
            
            number = random.random()
            if number < 0.15:
                choice = min(two_probsp)
                for i, prob in enumerate(transition_row):
                    if(prob == choice):
                        index = i
                        continue

            else:
                choice = max(two_probsp)
                for i, prob in enumerate(transition_row):
                    if(prob == choice):
                        index = i
                        continue

            current_state[state] = index  
        else:
            print('Not in List')
        
    return current_state