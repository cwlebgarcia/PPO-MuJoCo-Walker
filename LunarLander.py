import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create a vectorized training environment with continuous actions
# make_vec_env Parameters:
#   n_envs: The number of environments that run in parallel.
#   seed: Inital seed for the random number generator

# gym.make Parameters:
#   continuous: Lunar lander specific parameter to enable the continuous thruster control

# The lambda things acts as a function to return a unique lunar lander environment every time.
env = make_vec_env(lambda: gym.make("LunarLander-v3", continuous=True), n_envs=1)

# Initializes the PPO model to use a Multilayer Perceptron network for updating the policy and value functions,
# enables its environment, verbose enables the performance output messages after each run.
# Other parameters for the function include all of the hyperparameters for the PPO algorithm tutorial for cartpole plus more

# The model is an instance of the BaseRLModel class from stable-baselines3.
model = PPO("MlpPolicy", env, verbose=1)

# Load the model (optional)
# model = PPO.load("ppo_lunarlander_v3")

while 1:
    # Parameters for the learn method from the BaseRLModel class can be found in the documentation.
    # Maybe a useful one for the visualization stuff here that is possible with gym would be the callback parameter;
    # it is essentially an interrupt function that can be called with each step.
    # total_timesteps is the lower bound of number of samples to train on. Sounds a bit weird to me at the moment.
    # progress_bar also sounds like fun
    model.learn(total_timesteps=100)

    # Saves the parameters
    model.save("ppo_lunarlander_v3")

    # Create a rendering environment for evaluation.
    # We have a separate model to increase performance when we don't need to see the progress and also to act as a sort of testing stage.
    render_env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    obs, _ = render_env.reset()
    done = False
    total_reward = 0

    # Run the model on the test scenario
    while not done: 
        action, _ = model.predict(obs, deterministic=True) # Returns the models action and next state, given an observation. Selecting deterministic to true means that it will simply vhoose high highest action probability rather than sampling.
        obs, reward, terminated, truncated, _ = render_env.step(action) # Pass the action to the gymnasium environment 
        done = terminated or truncated # Ends the episode if the max number of steps is exceeded or otherwise
        total_reward += reward

    print(f"Total reward: {total_reward}")
    render_env.close()