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

# Set arrival rates of products
arrival_rates = [3, 3]

# Set order routing
routing = {
        'A': [0, 1],
        'B': [0, 1]
        }

# Set processing times per workstation (parts per hour)
process_rates = {
        'A': [7] * len(routing['A']),   
        'B': [7] * len(routing['B'])   
        }

# Set setup times for changing prodution type per workstation
setup_rates = {
        'A': [4] * len(routing['A']),   
        'B': [4] * len(routing['B'])    
        }

# Due dates
due_dates = {
        'A': 15,
        'B': 15
        }
