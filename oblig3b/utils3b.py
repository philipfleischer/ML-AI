import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display


def run_episode(environment, q_table, policy_function, update_function, alpha, gamma):
    """
    Runs an episode of an agent acting in an environment. This can be both according
    to SARSA or Q-learning.

    Note: Strictly speaking, Q-learning should choose the next action after the
    Q-table is updated, while SARSA chooses the next action before. In this implementation
    both algorithms choose the next action before updating the Q-table for simplicity.
    In practice this will not matter much. The details are covered in the first answer here:
    https://stackoverflow.com/questions/32846262/are-q-learning-and-sarsa-with-greedy-selection-equivalent

    Args:
        environment (gymnasium.wrappers.common.OrderEnforcing): A gymnasium environemnt.
        q_table (np.array): [n_states x n_actions]-shaped array of q-values.
        policy_function (callable): The policy function to use.
        update_function (callable): The learning rule to use.
        alpha (float): The learning rate.
        gamma (float): The discount factor.

    Returns:
        np.array: [n_states x n_actions]-shaped array of the updated q-values.
        float: The total reward accumulated during the episode.
    """
    state, info = environment.reset()
    done = False
    total_reward = 0

    # Choose first action
    action = policy_function(q_table=q_table, state=state)

    while not done:
        # Iterate environment
        next_state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated  # Terminated: Endstate reached, truncated: Max iterations reached
        total_reward += reward

        # Find next actiona and update q-table
        next_action = policy_function(q_table=q_table, state=next_state)
        transition = (state, action, reward, next_state, next_action)
        q_table = update_function(q_table=q_table, transition=transition, alpha=alpha, gamma=gamma)

        # Update states and actions for next step
        state = next_state
        action = next_action

    return q_table, total_reward


def train_agent(environment_name, n_episodes, policy_function, update_function, alpha, gamma, environment_args=None):
    """
    Trains a tabular reinforcement learning agent.
    Runs `n_episodes` amount of episodes of an agent using the `update_function` learning rule
    and `policy_function` policy function.

    Args:
        environment_name (str): Name of a gymnasium environment.
        n_episodes (int): The amount of episodes to run in the training
        policy_function (callable): The policy function to use.
        update_function (callable): The learning rule to use.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        environment_args (dict): Optional arguments to the environemnt.

    Returns:
        np.array: [n_states x n_actions]-shaped array of the updated q-values.
        np.array: [n_episodes]-shaped array of the accumulated rewards for each episodes.
    """
    if environment_args is None:
        environment_args = {"max_episode_steps": 100}
    environment = gym.make(environment_name, **environment_args)
    n_states = environment.observation_space.n
    n_actions = environment.action_space.n
    q_table = np.zeros((n_states, n_actions))
    rewards = np.zeros(n_episodes)

    for i in range(n_episodes):
        q_table, total_reward = run_episode(
            environment=environment,
            q_table=q_table,
            policy_function=policy_function,
            update_function=update_function,
            alpha=alpha,
            gamma=gamma,
        )
        rewards[i] = total_reward

    environment.close()
    return q_table, rewards


def visualize_episode(
    environment_name, q_table, policy_function, render_mode="rgb_array", delay=0.2, max_steps=30, environment_args=None
):
    """
    Visualizes a single episode using a trained q-table and a chosen policy.
    Works for both 'ansi' (text) and 'rgb_array' (images) render modes.

    Args:
        environment_name (str): Gymnasium environment name (for instance "CliffWalking-v1").
        q_table (np.ndarray): [n_states x n_actions] shaped array of trained Q-table.
        policy_function (callable): Policy used for action selection.
        render_mode (str): The visualization type used. "ansi" (text) or "rgb_array" (image).
        delay (float): Delay between frames in seconds.
        max_steps (int): Stops visualization after `max_steps` (or after an end-state is reached).
        environment_args (dict): Optional arguments to the environemnt.
    """
    if render_mode not in ("ansi", "rgb_array"):
        raise ValueError(f"Argument `render_mode` must be either `ansi` or `rgb_array`. Was {render_mode}. ")
    if environment_args is None:
        environment_args = {"max_episode_steps": 30}

    environment = gym.make(environment_name, render_mode=render_mode, **environment_args)
    state, info = environment.reset()
    total_reward = 0
    done = False

    if render_mode == "rgb_array":
        plt.figure(figsize=(6, 4))

    for step in range(max_steps):
        # Render output
        frame = environment.render()
        step_info = f"Step {step} | Total reward: {total_reward:.2f}"
        clear_output(wait=True)

        if render_mode == "ansi":
            print(frame)
            print(step_info)
        else:
            plt.imshow(frame)
            plt.axis("off")
            plt.title(step_info)
            display(plt.gcf())

        time.sleep(delay)

        # Iterate next step in episode
        action = policy_function(q_table=q_table, state=state)
        next_state, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        done = terminated or truncated

        if done:
            break
        state = next_state

    # Display last step
    clear_output(wait=True)
    last_step_info = f"Episode finished in {step + 1} steps. Total reward: {total_reward:.2f}"
    last_frame = environment.render()
    if render_mode == "ansi":
        print(last_frame)
        display(last_step_info)
    else:
        plt.imshow(last_frame)
        plt.axis("off")
        plt.title(last_step_info)
        display(plt.gcf())
        plt.close()

    environment.close()


def plot_rewards(rewards, title=None, show=True):
    """
    Plots the rewards over episodes.

    Args:
        rewards (np.array): [n_epoch]-shaped array of rewards.
        title (str, optional): Title to the plot.
        show (bool, optional): Whether or not to show the plot.
    """
    if title is None:
        title = "Loss over epochs"

    epochs = np.arange(1, len(rewards) + 1)
    plt.figure()
    plt.plot(epochs, rewards, marker="o", color="#3605A2")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if show is True:
        plt.show()
