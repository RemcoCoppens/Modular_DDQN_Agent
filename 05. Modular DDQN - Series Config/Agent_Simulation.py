""" Simulation using DDQN Agent """
import GC
from Simulation_Backend import FES, Event, Workstation, Order, Distribution, SimRes
from DDQN_Agent import Agent
from scipy import stats
import math
import numpy as np
from tqdm import tqdm

class AGENT_Simulation:
    def __init__(self, simTime, epsilon=1):
        # Set simulation parameters
        self.simulation_time = simTime
        self.nr_of_workstations = len(GC.process_rates[GC.product_list[0]])
        self.WS = [Workstation(station_nr=i, prod_list=GC.product_list) for i in range(self.nr_of_workstations)]
        self.fes = FES()
        self.simres = SimRes(sim_time=self.simulation_time, nr_of_ws=self.nr_of_workstations)
        self.agent = Agent(lr=5e-4,
                           gamma=0.99,
                           n_actions=3,
                           epsilon=epsilon,
                           eps_dec=3.5e-4,
                           eps_end=1e-2,
                           batch_size=32,
                           input_dims=14)
       
        # Initialize distributions
        self.arrival_dist = {p: Distribution(stats.expon(scale=1/GC.arrival_rates[i])) for (i, p) in enumerate(GC.product_list)}
        self.mfg_dist = {p: [Distribution(stats.expon(scale=1/GC.process_rates[p][i])) for i in range(self.nr_of_workstations)] for p in GC.product_list}
        self.setup_dist = {p: [Distribution(stats.expon(scale=1/GC.setup_rates[p][i])) for i in range(self.nr_of_workstations)] for p in GC.product_list}
        
        # Set runtime tracking variables
        self.order_nr = 0
        self.order_arrival = {p:[] for p in GC.product_list}
        self.indeces = []
        self.first_action= True
        
        # Initialize monitoring lists
        self.episodes = []
        self.performance = []
        self.utilization = []
        self.total_delay = []
        self.epsilon = []
        
        # Initialize tracking lists
        self.monitor_actions = [0] * len(GC.ACTION_NAMES)
        self.tracking_list_actions = []

    def run_sim(self, learn_agent=True):
        """ Run a single episode (run) of the simulation. """
        # Create orders and schedule first arrival event and add to fes for all product types
        for product_type in self.arrival_dist.keys():
            order = Order(num=self.order_nr, typ=product_type)
            self.order_nr += 1
            event = Event(created=0,
                          event_type=GC.OrderArr,
                          time=self.arrival_dist[product_type].rvs(), 
                          order=order,
                          loc=self.WS[order.steps[0]])
            self.fes.add(event)
        
        # Retrieve first event and set start time
        event = self.fes.next()
        time = event.time
        
        # Initialize action and index variable
        action = 0
        indx = 0
        
        while time < self.simulation_time:
            # "Cancel" order that are overdue by finishing them too late
            for ws in self.WS:
                late = [order for order in ws.queue if order.due_date < time]
                for late_order in late:
                    ws.queue.remove(late_order)
                    self.simres.record_finished_order(finished_order=late_order, 
                                                      cur_time=time)
            
            # Retrieve order object and set due date
            order = event.order
            
            # Handle order arrival event
            if event.type == GC.OrderArr:
                # Record order arrival
                self.order_arrival[order.type].append(time)
                
                # Set due date upon order arrival
                order.due_date = math.ceil(time + GC.due_dates[order.type])
                
                # Retrieve first destination of order
                destination_index = order.steps.pop(0)
                destination = self.WS[destination_index]
                
                # Record order arrival at workstation
                destination.record_arrival(time=time, 
                                           product_type=order.type)
                
                # Place order in workstation queue
                destination.queue.append(order)
                
                # If workstation is currently occupied, place order in queue
                if not destination.occupied:
                    if self.first_action:
                        # Initialize first state representation
                        state = self.agent.create_state_representation(time=time, 
                                                                       ws_nr=event.loc.station_nr, 
                                                                       WS=self.WS,
                                                                       order_arr=self.order_arrival)
                        
                        # Set first action to false
                        self.first_action = False
                        
                    else:                
                        # Retrieve state representation
                        state_ = self.agent.create_state_representation(time=time, 
                                                                        ws_nr=event.loc.station_nr, 
                                                                        WS=self.WS, 
                                                                        order_arr=self.order_arrival)
                        
                        # Set next state representation to current state representation
                        state = state_
                
                    # Choose action
                    action = self.agent.choose_action(observation=state)
                    self.monitor_actions[action] += 1
                        
                    # Take the chosen action, return if action was valid
                    _ = self.agent.take_action(time=time,
                                               action=action,
                                               workstation=destination,
                                               mfg_dist=self.mfg_dist,
                                               setup_dist=self.setup_dist,
                                               fes=self.fes)
                                                        
                # Create new order and corresponding arrival event and append to fes
                new_order = Order(num=self.order_nr, typ=order.type)
                self.order_nr += 1
                event = Event(created=time,
                              event_type=GC.OrderArr,
                              time=time + self.arrival_dist[order.type].rvs(),
                              order=new_order,
                              loc=self.WS[new_order.steps[0]])
                self.fes.add(event)
        
            # Handle finished process step event
            if event.type == GC.StepFinish:
        
                # Retrieve current order location
                location = event.loc
                
                # Record finished product
                location.record_finished(time=time,
                                         product_type=order.type)
        
                # check if order is finished
                if order.finished():
                    # Record finished order
                    self.simres.record_finished_order(finished_order=order, 
                                                      cur_time=time)

                else:
                    # Transport order to the next workstation
                    destination_index = order.steps.pop(0)
                    destination = self.WS[destination_index]
                    
                    # Record order arrival at workstation
                    destination.record_arrival(time=time,
                                               product_type=order.type)
                    
                    # Add order to destination queue
                    destination.queue.append(order)
                    
                    # If workstation is currently unoccupied, let agent decide upon action
                    if not destination.occupied:
                        # Retrieve state representation
                        state_ = self.agent.create_state_representation(time=time, 
                                                                        ws_nr=destination.station_nr, 
                                                                        WS=self.WS, 
                                                                        order_arr=self.order_arrival)
                        
                        # Set next state representation to current state representation
                        state = state_
                    
                        # Choose action
                        action = self.agent.choose_action(observation=state)
                        self.monitor_actions[action] += 1
                            
                        # Take the chosen action, return if action was valid
                        _ = self.agent.take_action(time=time,
                                                   action=action,
                                                   workstation=destination,
                                                   mfg_dist=self.mfg_dist,
                                                   setup_dist=self.setup_dist,
                                                   fes=self.fes)
 
                # Check if there are orders in the queue
                if len(location.queue) > 0:
                    # Retrieve state representation
                    state_ = self.agent.create_state_representation(time=time, 
                                                                    ws_nr=location.station_nr, 
                                                                    WS=self.WS, 
                                                                    order_arr=self.order_arrival)
                    
                    # Set next state representation to current state representation
                    state = state_
                    
                    # Choose action
                    action = self.agent.choose_action(observation=state)
                    self.monitor_actions[action] += 1
                        
                    # Take the chosen action, return if action was valid
                    _ = self.agent.take_action(time=time,
                                               action=action,
                                               workstation=location,
                                               mfg_dist=self.mfg_dist,
                                               setup_dist=self.setup_dist,
                                               fes=self.fes)
                
                # If no products in queue, workstation becomes idle (occupation set to false)
                else:
                    location.occupied = False
            
            # Retrieve next event from fes and set current time to event time
            event = self.fes.next()
            time = event.time
            
            # Record workstation utilization
            self.simres.record_workstation_utilization(Workstations=self.WS,
                                                       cur_time=time)
            
    def return_results(self):   
        """ Retrieve results from simres class. """
        # Calculate utilization rate per machine
        utilization = [round((wsu/self.simulation_time),1) for wsu in self.simres.ws_uptime]
        
        # Calculate percentage order on time delivered
        on_time_delivery = (self.simres.orders_on_time / self.simres.finished_orders) * 100 if self.simres.finished_orders != 0 else 0
        
        # Calculate total delay of all finished orders
        total_delay = self.simres.total_too_late
        
        # Return all variables
        return utilization, on_time_delivery, total_delay
    
    def reset(self, fname=None):
        """ Reset the environment. """
        self.fes = FES()
        self.WS = [Workstation(station_nr=i, prod_list=GC.product_list) for i in range(self.nr_of_workstations)]
        self.simres = SimRes(sim_time=self.simulation_time, nr_of_ws=self.nr_of_workstations)
        self.order_nr = 0
        self.order_arrival = {p:[] for p in GC.product_list}
        self.indeces = []
        self.first_action = True   
        self.monitor_actions = [0] * len(GC.ACTION_NAMES)
        
        self.agent = Agent(lr=5e-4,
                           gamma=0.99,
                           n_actions=3,
                           epsilon=0.01,
                           eps_dec=3.5e-4,
                           eps_end=1e-2,
                           batch_size=32,
                           input_dims=14)
        
        if fname != None:
            self.agent.load_model(file=fname)

    def run_episodes(self, nr_of_episodes, save_models=True, learn_agent=True, return_agg=False, fname=None):
        """ Run multiple episodes and return results. """
        # Loop over the number of episodes desired
        for i in tqdm(range(1, nr_of_episodes+1)):
            
            # Run the simulation
            self.run_sim(learn_agent=learn_agent)
            
            # Retrieve runtime statistics
            util, runtime_performance, delay = self.return_results()
            
            # Append and print episode number and its performance
            self.episodes.append(i)
            self.performance.append(runtime_performance)
            self.utilization.append(util)
            self.total_delay.append(delay)
            
#            # Print performance
#            print(f'Episode: {i} --> Performance: {round(runtime_performance, 2)} %  [Epsilon: {round(self.agent.epsilon, 4)}]')
            
            # Append action monitor to overall tracking list
            self.tracking_list_actions.append(self.monitor_actions.copy())
            
            # Reset simulation environment
            self.reset(fname=fname)  
        
        # Save final model if desired
        if save_models:
            self.agent.save_model(fname='Final_Model.h5')
        
        # If aggregates are desired, calculate and return mean and standard error
        if return_agg:
            mean_performance = np.mean(self.performance)
            SE_performance = np.std(self.performance) / np.sqrt(len(self.performance))
            return mean_performance, SE_performance