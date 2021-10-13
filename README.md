# The Maze Runner 
[![Build](https://github.com/mariadukmak/MazeRunner/actions/workflows/python-build.yml/badge.svg)](https://github.com/mariadukmak/MazeRunner/actions/workflows/python-build.yml)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)


This git repository houses the challenges for the course *Autonomy By Design*.

### [Jira Scrumboard](https://fancymazerunner.atlassian.net/jira/software/projects/DOOL/boards/1)

### Introduction Simulation
The simulation is made out of two parts, the `Environment` and the `Policy`.
The image below illustrates the simulation loop:

![](https://github.com/MariaDukmak/MazeRunner/blob/main/readme_assets/schets_simulatie.png)


### Installation
To install the package run the following command: 
```bash 
pip install .
```

If you are a developer you need to run this command:
```bash 
pip install .[dev]
```
### Multi agent
When adding the multi-agent aspect, we focused on the cooperation between the different agents.

At the end of each day (in the night) the agents who saved the day in the maze come back to the camp. 
These agents then share their explored maps with each other.

Then tasks are generated from the places where the agents have not been yet. These tasks are also distributed 
to the agents, and the agent properties (speed, memory death and policy) are taken into account during the distribution.

In the next day, the agent will try (depends on the policy) to complete the task given to him.

These aspects can be seen in the following files in the code:
* To split tasks at the end of each day: [Maze environment](https://github.com/MariaDukmak/MazeRunner/blob/main/mazerunner_sim/envs/mazerunner_env.py)
* The auction algoritme: [_auction_tasks](https://github.com/MariaDukmak/MazeRunner/blob/main/mazerunner_sim/envs/mazerunner_env.py)
* To include the task with the policy: [policies](https://github.com/MariaDukmak/MazeRunner/tree/main/mazerunner_sim/policies)

To understand how the agents thier actions decide, you can have a look at the following Flowchart:
[]()