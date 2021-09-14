""" Simulation using FIFO (Benchmark) """
import GC
from Simulation_Backend import FES, Event, Workstation, Order, Distribution, SimRes
from scipy import stats
import math
import numpy as np
from tqdm import tqdm


class FIFO_Simulation:
    def __init__(self, simTime):
        # Set simulation parameters
        self.simulation_time = simTime
        self.nr_of_workstations = len(GC.process_rates[GC.product_list[0]])
        self.WS = [Workstation(station_nr=i, prod_list=GC.product_list) for i in range(self.nr_of_workstations)]
        self.fes = FES()
        self.simres = SimRes(sim_time=self.simulation_time, nr_of_ws=self.nr_of_workstations)
        
        # Initialize distributions
        self.arrival_dist = {p: Distribution(stats.expon(scale=1/GC.arrival_rates[i])) for (i, p) in enumerate(GC.product_list)}
        self.mfg_dist = {p: [Distribution(stats.expon(scale=1/GC.process_rates[p][i])) for i in range(self.nr_of_workstations)] for p in GC.product_list}
        self.setup_dist = {p: [Distribution(stats.expon(scale=1/GC.setup_rates[p][i])) for i in range(self.nr_of_workstations)] for p in GC.product_list}
        
        # Set runtime tracking variables
        self.order_nr = 0
        self.order_arrival = {p:[] for p in GC.product_list}
        
        # Initialize monitoring lists
        self.episodes = []
        self.utilization = []
        self.performance = []
        self.total_delay = []
        
    def run_sim(self):
        """ Run single episode (run) of the manufacturing simulation """
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
        
        while time < self.simulation_time:                 
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
                                
                # If workstation unoccupied, start production of concerning order
                if not destination.occupied:
                    event = destination.start_production(order=order,
                                                         cur_time=time,
                                                         ev_time=time + self.mfg_dist[order.type][destination.station_nr].rvs(), 
                                                         setup_times=self.setup_dist)
                    # Add event to fes
                    self.fes.add(event)
                    
                # If workstation occupied, place order in queue
                else:
                    # Place order in workstation queue
                    destination.queue.append(order)
                
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
            else:
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
                    
                    # If workstation is currently occupied, place order in queue
                    if destination.occupied:
                        destination.queue.append(order)
                    
                    else:
                        event = destination.start_production(order=order,
                                                             cur_time=time,
                                                             ev_time=time + self.mfg_dist[order.type][destination.station_nr].rvs(), 
                                                             setup_times=self.setup_dist)
                        # Add event to fes
                        self.fes.add(event)
                
                # Check if there are orders in the queue
                if len(location.queue) > 0:
                    # Retrieve first order in the queue (FIFO)
                    next_order = location.queue.popleft()
                    
                    # Start production of the retrieved order
                    event = location.start_production(order=next_order,
                                                      cur_time=time,
                                                      ev_time=time + self.mfg_dist[order.type][destination.station_nr].rvs(),
                                                      setup_times=self.setup_dist)
                    # Add event to fes
                    self.fes.add(event)
                
                # If no other orders in the queue, set workstation occupation to False
                else:
                    location.occupied = False
            
            # Retrieve next event from fes and set current time to event time
            event = self.fes.next()
            time = event.time
            
            # Record workstation utilization
            self.simres.record_workstation_utilization(Workstations=self.WS,
                                                       cur_time=time)

    def return_results(self):
        """ Retrieve results from simRes class """
        # Calculate utilization rate per machine
        utilization = [round((wsu/self.simulation_time),1) for wsu in self.simres.ws_uptime]
        
        # Calculate percentage order on time delivered
        on_time_delivery = (self.simres.orders_on_time / self.simres.finished_orders) * 100
        
        # Calculate total delay of all finished orders
        total_delay = self.simres.total_too_late
        
        # Return all variables
        return utilization, on_time_delivery, total_delay

    def reset(self):
        """ Reset the environment """
        self.fes = FES()
        self.WS = [Workstation(station_nr=i, prod_list=GC.product_list) for i in range(self.nr_of_workstations)]
        self.simres = SimRes(sim_time=self.simulation_time, nr_of_ws=self.nr_of_workstations)
        self.order_nr = 0
        self.order_arrival = {p:[] for p in GC.product_list}
        
        
    def run_episodes(self, nr_of_episodes):
        """ Run multiple episodes and return results """
        # Loop over the number of episodes desired
        for i in tqdm(range(nr_of_episodes)):
            # Run the simulation
            self.run_sim()
            
            # Retrieve results from the executed simulation
            util, on_time, delay = self.return_results()
            
            # Append results to monitoring lists
            self.episodes.append(i)
            self.utilization.append(util)
            self.performance.append(on_time)
            self.total_delay.append(delay)
            
            # Reset simulation environment
            self.reset()
        
        # Calculate Confidence Interval (LCB, mean and UCB) of the performance
        LCB, mean, UCB = self.calculate_CI(self.performance)
        
        # Return CI values
        return LCB, mean, UCB
    
    def calculate_CI(self, values):
        """ Return mean and Confidence Intervals (CI) of the performance of the episodes """
        # Calculate mean and standard deviation of the given metric
        mean = np.mean(values)
        std = np.std(values)
        
        # Calculate the upper and lower control limit (UCB/LCB)
        UCB = mean + 1.96 * std
        LCB = mean - 1.96 * std
        
        # Return LCB, Mean and UCB
        return LCB, mean, UCB
        