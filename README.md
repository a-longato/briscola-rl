# Learning to Play Briscola with Reinforcement Learning
This is the repository for the coding project for the course of Reinforcement Learning, first year of the Master of Science in Data Science and Artificial Intelligence (a.y. 2025/2026, UniTS). The project has been developed by Alessandro Longato and Federico Cernaz.

## Overview
The objective of the project is to train and compare different agents learning to play Briscola. The trained agents are then pitted against each other in a round robin tournament. For more information about the algorithms and the results, refer to `presentation.pdf`.

*   **`iql.py`, `idqn.py`, `psro.py`**: These files contain the training scripts for the agents.
*   **`tournament.py`**: This file contains the script for the tournament.

## How to Run
First, clone this repository.
```sh
git clone https://github.com/a-longato/briscola-rl.git
cd briscola-rl
```
Then, install the required libraries. It is recommended to use a virtual environment.
```sh
pip install -r requirements.txt
```
To train each agent, run the specific training script.
