import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def CI_to_SE(performance, nr_of_episodes):
    """ Transform the (95%) confidence interval of FIFO to a SE """
    # Retrieve distance between bound and mean
    dist = performance[1] - performance[0]
    
    # Calculate standard deviation using distance
    stdev = dist / 1.96
    
    # Return standard error
    return stdev / np.sqrt(nr_of_episodes)
    

# Initialize performance dictionary
performance = {'FIFO_mean': [],
               'FIFO_se': [],
               'RANDOM_mean': [],
               'RANDOM_se': [],
               'STABLE_mean': [],
               'STABLE_se': [],
               'VARY_mean': [],
               'VARY_se': []}

# Retrieve all performance files in results folder
files = os.listdir(os.getcwd() + '/Results')

# Sort folders on increasing amount of steps
steps = [int(x.split('_')[0]) for x in files]
sorted_idx = np.argsort(np.array(steps))
files = [files[idx] for idx in sorted_idx]
steps = [steps[idx] for idx in sorted_idx]

# Loop through sorted files
for f in files:
    # Load data from pickle files
    file = open('Results/' + f, 'rb')
    data = pickle.load(file)
    
    # Sort data into seperate lists
    performance['FIFO_mean'].append(data['FIFO'][1])
    performance['FIFO_se'].append(CI_to_SE(performance=data['FIFO'], nr_of_episodes=1000))
    performance['RANDOM_mean'].append(data['Random'][0])
    performance['RANDOM_se'].append(data['Random'][1])
    performance['STABLE_mean'].append(data['Stable (best)'][0])
    performance['STABLE_se'].append(data['Stable (best)'][1])
    performance['VARY_mean'].append(data['Variable (final)'][0])
    performance['VARY_se'].append(data['Variable (final)'][1])

# Create plot of results
fig, ax = plt.subplots()
plt.title(f'Agent performance per number of workstations')

ax.set_xlabel('Number of parallel workstations')
ax.set_ylabel('Percentage Orders in Time [%]')
ax.errorbar(steps, performance['FIFO_mean'], performance['FIFO_se'], capsize=2, color='tab:green', label='FIFO performance')
ax.errorbar(steps, performance['RANDOM_mean'], performance['RANDOM_se'], capsize=2, color='tab:red', label='Random agent performance')
ax.errorbar(steps, performance['STABLE_mean'], performance['STABLE_se'], capsize=2, color='tab:blue', label='Stable agent performance')
ax.errorbar(steps, performance['VARY_mean'], performance['VARY_se'], capsize=2, color='tab:orange', label='Varying agent performance')
ax.tick_params(axis='y')

fig.tight_layout()
fig.subplots_adjust(top=0.88)
ax.legend()
plt.savefig('Results/Parallel_Errorbars.pdf')
    
