
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create a vectorized training environment with continuous actions
env = make_vec_env(lambda: gym.make("LunarLander-v3", continuous=True), n_envs=1)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

while 1:
    # Train the model
    model.learn(total_timesteps=100)

    # Save the model
    model.save("ppo_lunarlander_v3")

    # Load the model (optional)
    # model = PPO.load("ppo_lunarlander_v3")

    # Create a rendering environment for evaluation
    render_env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    obs, _ = render_env.reset()
    done = False
    total_reward = 0

    while not done: 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Total reward: {total_reward}")
    render_env.close()