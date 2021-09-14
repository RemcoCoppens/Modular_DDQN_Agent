from Agent_Simulation import AGENT_Simulation
from FIFO_Simulation import FIFO_Simulation
import pickle
import GC

def run_agent_sim(eps, time, fname=None):
    """ Run (trained) agent on environment for given number of episodes and return performance """
    # Initialize agent environment
    agent_sim = AGENT_Simulation(simTime=time, epsilon=0.01)
    
    # If learned agent is tested, load trained weights
    if fname != None:
        agent_sim.agent.load_model(file=fname)
    
    # Run agent and retrieve performance
    mean, se = agent_sim.run_episodes(nr_of_episodes=eps,
                                      save_models=False,
                                      learn_agent=False,
                                      return_agg=True,
                                      fname=fname)
    # Return agent performance
    return mean, se

# Set simulation parameters
simulation_time = 100
nr_eps_FIFO = 1000
nr_eps_AGENT = 100

# Initialize and run FIFO baseline
fifo_sim = FIFO_Simulation(simTime=simulation_time)
LCB, mean, UCB = fifo_sim.run_episodes(nr_of_episodes=nr_eps_FIFO)

# Retrieve performance of agent simulations
rnd_mean, rnd_se = run_agent_sim(eps=nr_eps_AGENT, time=simulation_time) # Random Agent
stab_mean, stab_se = run_agent_sim(eps=nr_eps_AGENT, time=simulation_time, fname='DDQN_static_best.h5') # Stable Best Agent
vary_mean, vary_se = run_agent_sim(eps=nr_eps_AGENT, time=simulation_time, fname='DDQN_dynamic_final.h5') # Vary Final Agent

# Integrate performance into dictionary
performance_dict = {'FIFO': [LCB, mean, UCB],
                    'Random': [rnd_mean, rnd_se],
                    'Stable (best)': [stab_mean, stab_se],
                    'Variable (final)': [vary_mean, vary_se]}

# Save data for later processing
file = open(f"Results/{len(GC.routing['A'])}_Step22222.pkl", "wb")
pickle.dump(performance_dict, file)
file.close()