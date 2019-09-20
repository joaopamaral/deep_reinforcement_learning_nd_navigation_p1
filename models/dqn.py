import torch
import os
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from tqdm.auto import tqdm

from models import Agent


def dqn(env: UnityEnvironment, n_episodes=2000, max_t=1000, eps_start=1.0,
        eps_end=0.01, eps_decay=0.995, seed=0, save_threshold=13.0):
    """
    Deep Q-Learning Training
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        seed (int): random seed
        save_threshold (float): value expected to save the model and exit training
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # get size of action and state
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    # create the agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=seed)

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in tqdm(range(1, n_episodes+1)):
        score = run_single_episode(env, brain_name, agent, max_t, eps, train_mode=True)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        print(score_log(i_episode, scores_window), end="")
        if i_episode % 100 == 0:
            if np.mean(scores_window) >= save_threshold:    # save if avg score is higher than the threshold
                print(score_log(i_episode, scores_window) + '\tSaved!')
                torch.save(agent.qnetwork_local.state_dict(),
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'results', 'checkpoint.pth'))
                # break
            else:
                print(score_log(i_episode, scores_window))

    return scores, agent


def run_single_episode(env: UnityEnvironment, brain_name, agent: Agent=None, max_t=1000, eps=0., train_mode=False):
    """
    Execute a single episode

    Params
    ======
        env (UnityEnvironment): enviroment
        brain_name (string): default brain name
        agent (Agent): agent that is responsible for control the actions (if no agent, a random action is chosen)
        max_t (int): max steps in each episode
        train_mode (bool): indicate if the environment is on the train mode

    Return
    ======
        score (float): total score of episode
    """
    env_info = env.reset(train_mode=train_mode)[brain_name]
    action_size = env.brains[brain_name].vector_action_space_size
    state = env_info.vector_observations[0]
    
    score = 0
    for _ in range(max_t):  # Run each step in episode
        action = agent.act(state, eps) if agent else np.random.randint(action_size)

        env_info = env.step(action)[brain_name]        # send the action to the environment

        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # get the done flag

        if agent and train_mode:
            agent.step(state, action, reward, next_state, done)
            
        state = next_state
        score += reward
        if done:    # Exit episode if done
            break

    return score


def score_log(i, s):
    """Log score"""
    return f'\rEpisode {i}\tAverage Score: {np.mean(s):.2f}'
