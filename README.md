# PPO for Walker in dm_control

This repository contains an implementation of Proximal Policy Optimization (PPO) for the `walker` environment from DeepMind's [dm_control](https://github.com/google-deepmind/dm_control) suite. The code is written in Python using PyTorch and demonstrates how to train a continuous control agent from scratch. This was a part of my coursework for **ESE 6500: Learning in Robotics**

## Motivation

The goal of this project is to provide a clear, minimal example of applying PPO to a physics-based continuous control task using the MuJoCo-powered dm_control suite. This serves as a practical reinforcement learning (RL) reference for continuous action spaces and can be adapted to other environments.

## Method

- **Algorithm:** Proximal Policy Optimization (PPO) with clipped surrogate objective.
- **Environment:** `walker/walk` from dm_control suite.
- **Neural Networks:** Actor and Critic are implemented using PyTorch modules (`uth_t`, `ValueFunction`).
- **Training Loop:** Collects trajectories, computes advantages using Generalized Advantage Estimation (GAE), and updates policy and value networks.
- **Key Features:**
  - Discounted cumulative rewards
  - Advantage normalization
  - Early stopping based on KL divergence
  - Supports GPU acceleration (if available)

## File Overview

| File         | Description                                                        |
|--------------|--------------------------------------------------------------------|
| `walker.py`  | Main PPO training script for dm_control walker environment         |
| `backend.py` | Contains neural network definitions for policy and value functions |

## Installation

1. **Install MuJoCo and dm_control**  
   Follow the [dm_control installation instructions](https://github.com/google-deepmind/dm_control#installation):

## Usage

To train a PPO agent on the walker environment, simply run:


- The script will automatically select GPU if available.
- Model weights are saved to `ppo_walk.pt` after training.

## Customization

- **Hyperparameters:**  
  Adjust `epochs`, `traj_steps`, learning rates, and other PPO parameters at the top of `walker.py`.
- **Environment:**  
  Change the environment by modifying the `suite.load('walker', 'walk', ...)` line.

## Attribution

- The dm_control suite is developed by [DeepMind](https://github.com/google-deepmind/dm_control)[3].
- PPO algorithm inspired by [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)[8].

## Credits

- Main PPO implementation: [Ethan Senatore]
- dm_control environment: DeepMind[3]
- See also: [PPO-PyTorch by nikhilbarhate99][2], [PPO-for-Beginners by ericyangyu][8]

