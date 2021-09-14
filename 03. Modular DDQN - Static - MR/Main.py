from Agent_Simulation import AGENT_Simulation
from FIFO_Simulation import FIFO_Simulation
import matplotlib.pyplot as plt
import numpy as np
import pickle 

#a_file = open("data.pkl", 'wb')
#pickle.dump(output, a_file)
#a_file.close()
#
b_file = open("data.pkl", 'rb')
output = pickle.load(b_file)

# Set simulation parameters
simulation_time = 100
nr_runs_FIFO = 1000
nr_runs_AGENT = 10
nr_eps_AGENT = 4000

# Initialize environments
fifo_sim = FIFO_Simulation(simTime=simulation_time)
agent_sim = AGENT_Simulation(simTime=simulation_time)

# Run episodes on simulation models
LCB, mean, UCB = fifo_sim.run_multiple(nr_of_episodes=nr_runs_FIFO)
agent_sim.run_multiple(nr_of_runs=nr_runs_AGENT, 
                       nr_of_episodes=nr_eps_AGENT,
                       verbose=True,
                       print_performance=False,
                       do_save_models=True)
# Retrieve output
output = agent_sim.performance

# Extract performance data from the monitoring dictionary
performance = np.array([output[k]['Performance'] for k in output.keys()])
utilization = np.array([output[k]['Utilization'] for k in output.keys()])
total_delay = np.array([output[k]['Total_Delay'] for k in output.keys()])

# Calculate standard error and mean performance
se_performance = np.std(performance, axis=0) / np.sqrt(nr_runs_AGENT)
se_utilization = np.std(utilization, axis=0) / np.sqrt(nr_runs_AGENT)
se_total_delay = np.std(total_delay, axis=0) / np.sqrt(nr_runs_AGENT)
mean_performance = np.mean(performance, axis=0)

# Create episodes and epsilon lists
episodes = list(range(1, nr_eps_AGENT + 1))
epsilon = [1 - 3.5e-4 * i if 1 - 3.5e-4 * i >= 1e-2 else 1e-2 for i in range(1, nr_eps_AGENT + 1)]
        
# Plot moving average graph
MA = [sum(mean_performance[i-2:i+3])/5 for i in range(2, len(mean_performance) - 2, 5)]
MA_SE = [sum(se_performance[i-2:i+3])/5 for i in range(2, len(se_performance) - 2, 5)]

MA_eps = [sum(epsilon[i-2:i+3])/5 for i in range(2, len(epsilon) - 2, 5)]
MA_episodes = list(range(2, len(mean_performance) - 2, 5))

# Transform data shape for plotting
LCB, mean, UCB = [LCB] * len(MA), [mean] * len(MA), [UCB] * len(MA)

# Create plot of results
fig, ax1 = plt.subplots()
plt.title(f'Agent Performance Moving Average (MA)')

color = 'tab:blue'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Percentage Orders in Time [%]', color=color)
ax1.plot(MA_episodes, UCB, color='tab:green', linestyle='--', label='UCB FIFO')
ax1.plot(MA_episodes, mean, color='tab:green', label='Mean FIFO')
ax1.plot(MA_episodes, LCB, color='tab:green', linestyle='--', label='LCB FIFO')
ax1.errorbar(MA_episodes, MA, MA_SE, capsize=2, color=color, label='Mean Agent Performance')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(MA_episodes, MA_eps, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
ax1.legend(loc='center left')
plt.savefig('Results/Stochastic_Env_Performance_MA5_ERRORBARS.pdf')








# Plot moving average graph
MA = [sum(mean_performance[i-4:i+5])/9 for i in range(4, len(mean_performance) - 4, 9)]
MA_SE = [sum(se_performance[i-4:i+5])/9 for i in range(4, len(se_performance) - 4, 9)]

MA_eps = [sum(epsilon[i-4:i+5])/9 for i in range(4, len(epsilon) - 4, 9)]
MA_episodes = list(range(4, len(mean_performance) - 4, 9))

# Transform data shape for plotting
LCB, mean, UCB = [LCB] * len(MA), [mean] * len(MA), [UCB] * len(MA)

# Create plot of results
fig, ax1 = plt.subplots()
plt.title(f'Agent Performance Moving Average (MA)')

color = 'tab:blue'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Percentage Orders in Time [%]', color=color)
ax1.plot(MA_episodes, UCB, color='tab:green', linestyle='--', label='UCB FIFO')
ax1.plot(MA_episodes, mean, color='tab:green', label='Mean FIFO')
ax1.plot(MA_episodes, LCB, color='tab:green', linestyle='--', label='LCB FIFO')
ax1.errorbar(MA_episodes, MA, MA_SE, capsize=2, color=color, label='Mean Agent Performance')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(MA_episodes, MA_eps, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
ax1.legend(loc='center left')
plt.savefig('Results/Stochastic_Env_Performance_MA9_ERRORBARS.pdf')