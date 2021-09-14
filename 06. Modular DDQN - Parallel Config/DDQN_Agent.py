# Import libraries
import numpy as np
import GC
import os
import torch
import torch.nn as nn


class ReplayBuffer():
    """
    Enabling the agent to sample non correlated samples from the data, as 1 step TD is highly unstable
    """
    def __init__(self, max_size, input_shape):
        # Set max memory size and initialize counter to 0
        self.mem_size = max_size
        self.mem_cntr = 0
        
        # Initialize memory lists
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        
    def store_transition(self, state, action, state_):
        """ Store state information in memory """
        # Retrieve position of the first available spot to save information
        index = self.mem_cntr % self.mem_size
        
        # Save memory to the given index point
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action

        # Increment memory counter
        self.mem_cntr += 1 
        
        # Return insertion index
        return index

    def sample_buffer(self, batch_size):
        """ Prevent sample contains zero values, only sample filled memory """
        # Retrieve the index to which the reward is filled
        max_mem = len(self.reward_memory)

        # Sample a batch size amount of distinct random values from the range of filled indexes
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # Retrieve information from sampled batch indexes
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        # Return batch information
        return states, actions, rewards, new_states

    
class DDQNetwork(nn.Module):
    """ Class containing the Dueling Deep Q Network, able to retrieve both state and advantage values. """
    def __init__(self, n_actions, input_dims, fc1_dims, fc2_dims):
        super(DDQNetwork, self).__init__()
        self.lin_1 = nn.Linear(input_dims, fc1_dims)
        self.lin_2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)
    
    def call(self, state, req_grad=True):
        """ Conduct forward pass of the features through the network, return Q-value """
        # Convert state to tensor
        state_T = torch.Tensor(state)
        
        # For training, keep track of gradients
        if req_grad:
            # Take the features through the two dense layers
            x = self.lin_1(state_T)
            x = self.lin_2(x)
            
            # Take output of these dense layers through A layer seperately
            V = self.V(x)
            A = self.A(x)
            
            # Calculate Q value by adding advantage to value function
            Q = (V + (A - torch.mean(A)))
            
            # Return the calculated Q value
            return Q
        
        # Else set torch to no grad
        else:
            # Set NN to no grad, as only forward pass is required
            with torch.no_grad():
                # Take the features through the two dense layers
                x = self.lin_1(state_T)
                x = self.lin_2(x)
                
                # Take output of these dense layers through A layer seperately
                V = self.V(x).numpy()
                A = self.A(x).numpy()
                
                # Calculate Q value by adding advantage to value function
                Q = (V + (A - np.mean(A)))
                
                # Return the calculated Q value
                return Q
    
    def advantage(self, state):
        """ Conduct forward pass only for the advantage values """
        # Convert state to tensor
        state_T = torch.Tensor(state)
        
        # Set NN to no grad, as only forward pass is required
        with torch.no_grad():
            # Take the features through the two dense layers
            x = self.lin_1(state_T)
            x = self.lin_2(x)
            
            # Take output of these dense layers through both the A layer
            A = self.A(x).numpy()
            
            # Return the advantage value
            return A    
    

class Agent(): 
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, eps_dec=5e-6, eps_end=1e-2,
                 mem_size=100000, fname='DDQN.h5', fc1_dims=32,
                 fc2_dims=64, replace=100):
        # Initiate agent characteristics
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.fname = fname
        self.replace = replace
        self.batch_size = batch_size
        
        # Initialize Online Evaluation Network
        self.q_eval = DDQNetwork(n_actions, input_dims, fc1_dims, fc2_dims)

        # Initialize Target Network for action selection
        self.q_next = DDQNetwork(n_actions, input_dims, fc1_dims, fc2_dims)
        
    def load_model(self, file=None):
        """ Load the model weights from the given file name """
        if file == None:
            self.q_eval.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model Saves', self.fname)))
            self.q_next.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model Saves', self.fname)))
        else:
            self.q_eval.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model Saves', file)))
            self.q_next.load_state_dict(torch.load(os.path.join(os.getcwd() + '/Model Saves', file)))
        
    def choose_action(self, observation):
        """ Choose action (epsilon-greedily) based on the observation of the current state of the environment  """
        # If random value is below epsilon, choose random action
        if np.random.random() < self.epsilon:
            action= np.random.choice(self.action_space)
        
        # Else take a greedy action
        else:
            # Convert state observation to useable format and retrieve action advantage values from network
            state = np.array([observation])
            actions = self.q_eval.advantage(state)

            # Take action with the highest advantage value
            action = np.argmax(actions)

        # Return the selected action
        return action     
        
    def create_state_representation(self, time, ws_nr, WS, order_arr, queue):
        """ 
        Generate state representation: 
            - queue_amounts: Amount of orders in the queue per product class (integer)
            - order_arr: The average inter-arrival rate of order per product class (float)
            - inter_arr: The average inter-arrival rate at the workstation per product class (float)
            - in_machine: Represents the product class currently being processed (binary)
            - min_slack: The shortest amount of time to due date per product class (float)
            - avg_process: Average observed processing time per product class (float)
            - avg_setup: Average observed setup time per product class (float)
        """
        # Retrieve the remaining processing time per product class in queue (if not produced before, return 0)
        rem_process_time = {'A': 0.164780618776484, 'B': 0.164780618776484}
            
        # Create dictionary of due dates per product class present in the workstation queue
        dd_dict = {prod:[] for prod in GC.product_list}  
        for order in queue:
            dd_dict[order.type].append(order.due_date)
        
        # Calculate the amount of orders per product class in the queue and convert to percentage of total
        queue_amounts = [len(dd_dict[p]) for p in GC.product_list]
        total_queue = sum(queue_amounts)
        queue_amounts = [x/total_queue if total_queue > 0 else 0 for x in queue_amounts]
        
        # Convert order arrival rates to percentage of total
        total_order_arr = sum([sum(x) if len(x) > 0 else 0 for x in order_arr.values()])
        order_arr = [sum(order_arr[k])/total_order_arr if len(order_arr[k]) > 0 else 0 for k in order_arr.keys()]
        
        # Calculate the minimal slack (time till due date - remaining process time) per product class and convert to percentage of total
        min_slack = [min(dd_dict[p]) - rem_process_time[p] - time if len(dd_dict[p]) > 0 else 0 for p in GC.product_list] 
        total_min_slack = sum(min_slack)
        try:
            min_slack = [x/total_min_slack for x in min_slack]
        except ZeroDivisionError:
            print(dd_dict)
        
        # Calculate average inter-arrival, processing and setup times per product class for the concerning workstation
        inter_arr = [np.mean(np.diff(WS[ws_nr].inter_arrival[p])) if len(WS[ws_nr].inter_arrival[p]) > 1 else 0 for p in GC.product_list]
        avg_process = [np.mean(np.diff(WS[ws_nr].processing_times[p])) if len(WS[ws_nr].processing_times[p]) > 1 else 0 for p in GC.product_list]
        avg_setup = [np.mean(WS[ws_nr].setup_times[p]) if len(WS[ws_nr].setup_times[p]) > 0 else 0 for p in GC.product_list]
        
        # Convert amounts to percentages of total
        total_inter_arr = sum(inter_arr)
        total_avg_process = sum(avg_process)
        total_avg_setup = sum(avg_setup)
        inter_arr = [x/total_inter_arr if total_inter_arr > 0 else 0 for x in inter_arr]
        avg_process = [x/total_avg_process if total_avg_process > 0 else 0 for x in avg_process]
        avg_setup = [x/total_avg_setup if total_avg_setup > 0 else 0 for x in avg_setup]
        
        # Retrieve the product class of the workstation
        in_machine = [1 if x == WS[ws_nr].prod_type else 0 for x in GC.product_list]
        
        # Return state representation
        return np.array([queue_amounts, 
                         order_arr, 
                         inter_arr, 
                         in_machine,
                         min_slack,
                         avg_process,
                         avg_setup]).flatten()
            
    def take_action(self, time, action, workstation, mfg_dist, setup_dist, fes, queue):
        """ 
        Execute the given action.
        """
        # Handle action 1: "Produce product A"
        if action == 0:
            # Retrieve orders of type A from workstation queue
            orders = [x for x in queue if x.type == 'A']
            
            # Check if type A products are present in queue
            if len(orders) > 0:
                # Pop the first order in queue from the list and remove from queue
                order = orders.pop(0)
                queue.remove(order)
                
                # Start production for the given order
                event = workstation.start_production(order=order,
                                                     cur_time=time,
                                                     ev_time=time + mfg_dist[order.type][workstation.station_nr].rvs(),
                                                     setup_times=setup_dist) 
                # Append to fes
                fes.add(event)
                
                # Return positive validity check for action
                return True
            else:
                # Set workstation occupation to False
                workstation.occupied = False
                
                # Return negative validity check for action
                return False
        
        # Handle action 2: "Produce product B"
        if action == 1:
            # Retrieve orders of type B from workstation queue
            orders = [x for x in queue if x.type == 'B']
            
            # Check if type A products are present in queue
            if len(orders) > 0:
                # Pop the first order in queue from the list and remove from queue
                order = orders.pop(0)
                queue.remove(order)
                
                # Start production for the given order
                event = workstation.start_production(order=order,
                                                     cur_time=time,
                                                     ev_time=time + mfg_dist[order.type][workstation.station_nr].rvs(),
                                                     setup_times=setup_dist) 
                # Append to fes
                fes.add(event)                 
                
                # Return positive validity check for action
                return True
            else:
                # Set workstation occupation to False
                workstation.occupied = False
                
                # Return negative validity check for action
                return False
        
        # Handle action 3: "Wait, stay idle"
        if action == 2:
            # Set workstation occupation to False
            workstation.occupied = False
            
            # Return positive validity check for action
            return True
        
    def retrieve_memory(self):
        """ Return the agent memory """
        # Retrieve Agent Memory
        state_memory = self.memory.state_memory
        new_state_memory = self.memory.new_state_memory
        action_memory = self.memory.action_memory
        reward_memory = self.memory.reward_memory
        
        return state_memory, new_state_memory, action_memory, reward_memory
