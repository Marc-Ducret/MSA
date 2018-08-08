# Project Architecture

Agents are implemented as a combination of Java and Python.

## Environment

Environments or tasks should be described in Java, by extending the Environment class. Those define the observations, actions and rewards of the agent. For now, each agent has its own environment instance.

## Learning Algorithms

The learning algorithms should be written in Python, they interface with Java through a `MinecraftEnv` instance that implements the [gym](https://gym.openai.com/) interface. This environment assumes that `env.reset()` is called initially and after each step that returns `done = True`.

## Communication between Java and Python

Python scripts communicate with the Minecraft server through TCP sockets.

## Examples

* `agent_dummy` and `agent_random` are the most simple agent implementations
* `agent_tree` is an hardcoded agent for the `Pattern` environment
* `agent_ppo` and `agent_acktr` are wrappers for [OpenAI's baselines](https://github.com/VengeurK/Villagers-Baselines-Fork)
* `agent_rl` runs [PyTorch RL algorithms](https://github.com/VengeurK/pytorch-a2c-ppo-acktr)
* `agent_nearest` is a K-Neareast-Neighbor agent
* `test_controller` is a simple usage of `MinecraftController`
* `multi_agent_test_pattern` runs multiple agents in a single environment with only one Python process
* `agent_imitation_train` does imitation learning
