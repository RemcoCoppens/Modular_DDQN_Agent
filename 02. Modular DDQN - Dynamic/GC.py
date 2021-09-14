""" Global Constants """
from scipy import stats

# Set DEBUG MODE on/off
DEBUG_MODE = False

# Set event types
OrderArr = 0
StepFinish = 1
EVENT_NAMES = ['Order Arrival', 'Finished Product']

# Set action names
ACTION_NAMES = ['Produce A', 'Produce B', 'Stay Idle']

# Set product list
product_list = ['A', 'B']

# Set arrival rates of products
arrival_rates = [3, 3]
arrival_unif = stats.uniform(1, 6)

# Set order routing
routing = {
        'A': [0],
        'B': [0]
        }

# Set processing times per workstation (parts per hour)
process_rates = {
        'A': [6],   
        'B': [6]    
        }
process_unif = {
        'A': stats.uniform(3, 9),
        'B': stats.uniform(3, 9)        
        }

# Set setup times for changing prodution type per workstation
setup_rates = {
        'A': [2],
        'B': [2]    
        }
setup_intervals = {
        'A': stats.uniform(2, 6),
        'B': stats.uniform(2, 6)
        }

# Due dates
due_dates = {
        'A': 15,
        'B': 15
        }


