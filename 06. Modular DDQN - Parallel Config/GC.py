""" Global Constants """

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

# Set number of workstations
nr_of_ws = 10

# Set arrival rates of products
arrival_rates = [3, 3]

# Set order routing
routing = {
        'A': [0],
        'B': [0]
        }

# Set processing times per workstation (parts per hour)
process_rates = {
        'A': [0.55] * nr_of_ws,   
        'B': [0.55] * nr_of_ws    
        }

# Set setup times for changing prodution type per workstation
setup_rates = {
        'A': [2] * nr_of_ws,
        'B': [2] * nr_of_ws    
        }

# Due dates
due_dates = {
        'A': 15,
        'B': 15
        }
