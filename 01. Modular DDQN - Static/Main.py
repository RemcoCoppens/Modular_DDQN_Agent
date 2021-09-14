from Agent_Simulation import AGENT_Simulation
from FIFO_Simulation import FIFO_Simulation
import matplotlib.pyplot as plt
import numpy as np

# Set simulation parameters
simulation_time = 100
nr_eps_FIFO = 1000
nr_eps_AGENT = 4000

# Initialize environments
fifo_sim = FIFO_Simulation(simTime=simulation_time)
agent_sim = AGENT_Simulation(simTime=simulation_time)

# Run episodes on simulation models
LCB, mean, UCB = fifo_sim.run_episodes(nr_of_episodes=nr_eps_FIFO)
agent_sim.run_episodes(nr_of_episodes=nr_eps_AGENT)

# Retrieve data from class object
episodes = agent_sim.episodes
performance = agent_sim.performance
epsilon = agent_sim.epsilon



############# Plotting performance #############



# Plot moving average graph
MA = [sum(performance[i-2:i+3])/5 for i in range(2, len(performance) - 2, 5)]
MA_eps = [sum(epsilon[i-2:i+3])/5 for i in range(2, len(epsilon) - 2, 5)]
MA_episodes = list(range(2, len(performance) - 2, 5))

# Transform data shape for plotting
LCB_adj, mean_adj, UCB_adj = [LCB] * len(MA), [mean] * len(MA), [UCB] * len(MA)

# Create plot of results
fig, ax1 = plt.subplots()
plt.title(f'Agent Performance Moving Average (MA)')

color = 'tab:blue'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Percentage Orders in Time [%]', color=color)
ax1.plot(MA_episodes, UCB_adj, color='tab:green', linestyle='--' , label='FIFO UCB')
ax1.plot(MA_episodes, mean_adj, color='tab:green', label='FIFO MEAN')
ax1.plot(MA_episodes, LCB_adj, color='tab:green', linestyle='--', label='FIFO LCB')
ax1.plot(MA_episodes, MA, color=color, label='Agent Performance')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(MA_episodes, MA_eps, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
ax1.legend(loc='center left')
plt.savefig('Results/Stochastic_Env_Performance_MA5.pdf')





# Plot moving average graph
MA = [sum(performance[i-4:i+5])/9 for i in range(4, len(performance) - 4, 9)]
MA_eps = [sum(epsilon[i-4:i+5])/9 for i in range(4, len(epsilon) - 4, 9)]
MA_episodes = list(range(4, len(performance) - 4, 9))

# Transform data shape for plotting
LCB_adj, mean_adj, UCB_adj = [LCB] * len(MA), [mean] * len(MA), [UCB] * len(MA)

# Create plot of results
fig, ax1 = plt.subplots()
plt.title(f'Agent Performance Moving Average (MA)')

color = 'tab:blue'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Percentage Orders in Time [%]', color=color)
ax1.plot(MA_episodes, UCB_adj, color='tab:green', linestyle='--' , label='FIFO UCB')
ax1.plot(MA_episodes, mean_adj, color='tab:green', label='FIFO MEAN')
ax1.plot(MA_episodes, LCB_adj, color='tab:green', linestyle='--', label='FIFO LCB')
ax1.plot(MA_episodes, MA, color=color, label='Agent Performance')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(MA_episodes, MA_eps, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
ax1.legend(loc='center left')
plt.savefig('Results/Stochastic_Env_Performance_MA9.pdf')







############# Plotting policy #############

# Retrieve action information
actions = agent_sim.tracking_list_actions
actions_perc = [[a[0]/sum(a), a[1]/sum(a), a[2]/sum(a)] for a in actions]

# Convert action information to useful forman
actions_A = [a[0] for a in actions_perc]
actions_B = [a[1] for a in actions_perc]
actions_I = [a[2] for a in actions_perc]

# Create plot of policy
fig, ax = plt.subplots()
plt.title(f'Agent Performance Moving Average (MA)')

ax.set_ylabel('Policy Action Selection')
ax.bar(list(range(len(actions_A))), actions_A, color='tab:red', alpha=0.5, label='Produce A')
ax.bar(list(range(len(actions_A))), actions_B, bottom=actions_A, color='tab:green', alpha=0.5, label='Produce B')
ax.bar(list(range(len(actions_A))), actions_I, bottom=np.array(actions_A)+np.array(actions_B), color='tab:blue', alpha=0.5, label='Stay Idle')
ax.tick_params(axis='y')

fig.tight_layout()
fig.subplots_adjust(top=0.88)
ax.legend(loc='center left')
plt.savefig('Results/Stochastic_Env_Performance_ACTIONS.pdf')



