# Modular_DDQN_Agent
This repository shows the code used for the research into a modular framework for a DRL implementation. A modular framework implies that the agent can only see its direct environment, which enables the duplication of agents in either series and/or parallel configurations

Folder 01. Modular DDQN - Static:
  This folder represents one of the environments in which the DDQN agent is trained. 
  Through static environmental, meaning the settings stay the same for all episodes (simulation runs), the agent is focussed on a single environment setting.

Folder 02. Modular DDQN - Dynamic:
  This folder represents the second environment in which the DDQN agent is trained.
  Training using a dynamic environment implies changing environmental settings over consecutive episodes (simulation runs).
  It was expected that this would increase the robustness, and thus performance, of the agent on the parallel and serial tasks.
 
Folder 03. Modular DDQN - Static - MR:
  This folder shows an identical implementation as folder 1, but instead of running a single training trajectory it runs multiple.
  The information gained from this is used to determine the confidence intervals of the learning trajectory of the agent.

Folder 04. Modular DDQN - Series Config:
  This folder represents the first test environment. 
  In this environment the performance of three agents (statically/dynamically trained and a random agent) is tested in a serial configuration.
  What is shown is that the static agent keeps its performance on a process of up to 10 serial process steps.

Folder 05. Modular DDQN - Parallel Config:
  This folder represents the second test environment. 
  In this environment the performance of three agents (statically/dynamically trained and a random agent) is tested in a parallel configuration.
  What is shown is that the static agent does not keep its performance well enough on a process of up to 10 parallel process steps, but still outperforms the random agent.
  An improvement could be attained by training the agent on a small serial and parallel configuration to enable inclusion of environmental values in the state representation of the agent.
