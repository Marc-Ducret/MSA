# Project Architecture

Agents are implemented as a combination of Java and Python.

## Environment

Environments or tasks should be described in Java extending the Environment class. Those define the observations, actions and rewards of the agent. For now, each agent has its own environment instance.

## Learning Algorithms

The learning algorithms should be written in Python, they interface with Java through a `MinecraftEnv` instance that implements the [gym](https://gym.openai.com/) interface. This environment assumes that `env.reset()` is called initially and after each step that returns `done = True`.

## Communication between Java and Python

Each agent has its own Python process. Each of those process communicates with Minecraft through its `stdin` and `stdout`. Also, `stderr` can be used to display debugging information.
