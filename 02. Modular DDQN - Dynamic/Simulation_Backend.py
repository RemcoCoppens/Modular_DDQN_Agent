import heapq
from collections import deque
import GC

class FES:
    """
    The Future Event Set (FES) holds the 'planned' events in chronological order.
    """
    def __init__(self):
        self.events = []

    def __str__(self):
        """ Print current events present in the Future Event Set """
        message = f'The FES currently contains {len(self.events)} events, namely: \n'
        sortedEvents = sorted(self.events)
        for event in sortedEvents:
            message += '  ' + str(event) + '\n'
        return message

    def add(self, event):
        """ Add event to the FES """
        heapq.heappush(self.events, event)

    def next(self):
        """ Retrieve next event in the FES """
        return heapq.heappop(self.events)

    def is_empty(self):
        """ Return if empty priority queue """
        return len(self.events) == 0


class Event:
    """
    Create events for the Future Event Set (FES).
    """
    def __init__(self, created, event_type, time, order, loc=None):
        self.created = created
        self.type = event_type
        self.time = time
        self.order = order
        self.loc = loc

    def __str__(self):
        """ Print event information """
        if self.type == GC.OrderArr:
            return f'---{GC.EVENT_NAMES[self.type]} (Product: {self.order.type})---\n' \
                   f'\t Execution time: {str(round(self.time, 3))} \n' \
                   f'\t Location: Station {self.loc.station_nr}'
        else:
            return f'---{GC.EVENT_NAMES[self.type]} (Location: {self.loc.station_nr})---\n' \
                   f'\t Execution time: {str(round(self.time, 3))} \n' \
                   f'\t Location: Station {self.loc.station_nr}'

    def __lt__(self, other):
        """ Check chronological order of event to other events """
        return self.time < other.time


class Workstation:
    """
    A workstation is one step of the manufacturing process.
    """
    def __init__(self, station_nr, prod_list):
        self.station_nr = station_nr
        self.occupied = False
        self.prod_type = None
        self.queue = deque()
        self.start_process_time = 0
        self.inter_arrival = {prod:[] for prod in prod_list}
        self.processing_times = {prod:[] for prod in prod_list}
        self.setup_times = {prod:[] for prod in prod_list}
    
    def record_arrival(self, time, product_type):
        """ Record order arrival at workstation """
        self.inter_arrival[product_type].append(time)
        
    def record_finished(self, time, product_type):
        """ Record the process time of the finished product """
        self.processing_times[product_type].append(time)
    
    def cancel_overdue_orders(self, time, simres):
        """ Cancel orders that are overdue and record them as 'finished' but not on time """
        # Loop through all orders in the queue of the workstation
        orders_to_remove = []
        for order in self.queue:
            
            # If order is overdue
            if order.due_date < time:
                # Document finished order (that is too late)
                simres.record_finished_order(finished_order=order,
                                             cur_time=time) 
                
                # Keep track of orders to remove
                orders_to_remove.append(order)
        
        # Loop over all orders to remove and remove them from the queue
        for order in orders_to_remove:
            self.queue.remove(order)
        
    def start_production(self, order, cur_time, ev_time, setup_times):
        """ Set workstation to occupied and create a finishing event """
        # Set workstation to occupied
        self.occupied = True
        
        # If setup is needed, increment event time
        if self.prod_type != order.type:
            # Retrieve setup time from distribution
            setup_time = setup_times[order.type][self.station_nr].rvs()
            
            # Append setup time to setup time memory
            self.setup_times[order.type].append(setup_time)
            
            # Create finish event with setup time
            event = Event(created=cur_time,
                          event_type=GC.StepFinish,
                          time=ev_time + setup_time,
                          order=order,
                          loc=self)
            
            # Register time processing starts (after setup)
            self.start_process_time = cur_time + setup_time
            
            # Set production type of workstation to order product type
            self.prod_type = order.type
        
        else:
            # Create finish event without setup time
            event = Event(created=cur_time,
                          event_type=GC.StepFinish,
                          time=ev_time,
                          order=order,
                          loc=self)
            
            # Register time processing starts
            self.start_process_time = cur_time

        # Return finish event
        return event

    def __str__(self):
        """ Print station information """
        return f'Station {self.station_nr} currently has {len(self.queue)} orders waiting'


class Order:
    """
    An order is a set of processing steps to be executed
    """
    def __init__(self, num, typ, due_date=0):
        self.num = num
        self.type = typ
        self.due_date = due_date
        self.steps = GC.routing.get(self.type).copy()
    
    def __str__(self):
        return f'Order type: {self.type}, with due date: {self.due_date} \n Remaining routing: {self.steps}'
    
    def finished(self):
        return len(self.steps) == 0


class SimRes:
    """
    Record simulation results
    """
    def __init__(self, sim_time, nr_of_ws):
        self.sim_time = sim_time
        self.ws_uptime = [0] * nr_of_ws 
        self.prev_time = 0
        self.finished_orders = 0
        self.orders_on_time = 0
        self.total_too_late = 0
    
    def record_finished_order(self, finished_order, cur_time):
        """ Record a finished order """
        # Increment finished orders counter
        self.finished_orders += 1
        
        # Retrieve order due date
        due_date = finished_order.due_date
        
        # If order is finised on time, increment counter
        if due_date >= cur_time:
            self.orders_on_time += 1
        # else calculate deviation and add to total_too_late (loss)
        else:
            self.total_too_late += (due_date - cur_time)
    
    def record_workstation_utilization(self, Workstations, cur_time):
        """ At every event register workstation utilization """
        # Calculate time between event and previous event
        inter_event_time = cur_time - self.prev_time
        
        # Loop over all workstations
        for i, ws in enumerate(Workstations):
            # If the workstation was occupied, add inter event time
            if ws.occupied:
                self.ws_uptime[i] += inter_event_time
            
        # Set previous time as current time
        self.prev_time = cur_time
    
    def __str__(self):
        """ Print simulation results """
        message = f'--- Workstation Utilization: ---\n'
        for i, ws in enumerate(self.ws_uptime):
            message += f'\t - WS{i}: {round((ws/self.sim_time * 100), 1)}% \n' 
        message += f'\n Orders on time: {round((self.orders_on_time/self.finished_orders * 100), 1)}%, with total delay (loss): {round(self.total_too_late, 2)}'
        
        return message

class Distribution:
    """
    Speed up the retrieval of taking random variates from the given distribution by taking multiple simultaneously.
    """

    n = 10000  # standard random numbers to generate

    def __init__(self, dist):
        self.dist = dist
        self.resample()

    def __str__(self):
        return str(self.dist)

    def resample(self):
        self.randomNumbers = self.dist.rvs(self.n)
        self.idx = 0

    def rvs(self, n=1):
        '''
        A function that returns n (=1 by default) random numbers from the specified distribution.

        Returns:
            One random number (float) if n=1, and a list of n random numbers otherwise.
        '''
        if self.idx >= self.n - n:
            while n > self.n:
                self.n *= 10
            self.resample()
        if n == 1:
            rs = self.randomNumbers[self.idx]
        else:
            rs = self.randomNumbers[self.idx:(self.idx + n)]
        self.idx += n
        return rs
