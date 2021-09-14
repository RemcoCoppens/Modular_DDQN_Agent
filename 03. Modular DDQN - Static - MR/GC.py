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
        'A': [0],
        'B': [0]
        }

# Set processing times per workstation (parts per hour)
process_rates = {
        'A': [6],   
        'B': [6]    
        }

# Set setup times for changing prodution type per workstation
setup_rates = {
        'A': [2],
        'B': [2]    
        }

# Due dates
due_dates = {
        'A': 15,
        'B': 15
        }


