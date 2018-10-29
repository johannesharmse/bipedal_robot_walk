# Credit to Colin Skow
# Adapted code from https://github.com/colinskow/move37/blob/master/ars/ars.py

import gym
from gym import wrappers
import numpy as np
import os


class Normalizer():
    """Normalize input values.
    Args:
        n_inputs: Number of input values
    """

    def __init__(self, n_inputs):
        self.n = np.zeros(n_inputs)
        self.mean = np.zeros(n_inputs)
        self.mean_diff = np.zeros(n_inputs)
        self.var = np.zeros(n_inputs)

    def observe(self, x):
        """
        Update mean and variance values based on observation values.

        Args:
            x(Numpy array): Observed value
        """
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        """
        Normalize input values.

        Args:
            inputs(list of floats): Input values to normalize.

        Returns:
            Normalized input values with same shape and `inputs`.
        """
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
        

class Policy():
    """Policy Agent should follow
    
    Args: 
        input_size(int): Number of weights in input layer.
        output_size(int): Number of weights in output layer.
    """

    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))

    def evaluate(self, input, noise, delta=None, direction=None):
        """
        Add noise to weights based on gradient decent direction 
        and return new theta values dot product (with input).

        Args:
            input(list of floats): Normalized observed input values.
            delta(list of floats): Relative step size (multiplied with noise) during gradient descent
            noise(float): Amount of delta distortion to add to weights.
        
        Returns:
            N-dimensional array of dot product of input and 
            new theta values.
        """

        if direction is None:
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + noise * delta).dot(input)
        elif direction == "-":
            return (self.theta - noise * delta).dot(input)
        
    def sample_deltas(self, n_deltas):
        """Sample random delta values from normal distribution for each input weight.

        Args:
            n_deltas(int): Number of deltas to sample for each input weight.
        
        Returns:
            List of lists of floats.
        """

        return [np.random.randn(*self.theta.shape) for _ in range(n_deltas)]

    def update(self, rollouts, lr, sigma_rewards):
        """
        Update theta values / weights based on learning rate and step size.

        Args:
            lr(float): Learning Rate
            rollouts(list of tuples): Positive rewards, Negative rewards, Delta values.
            sigma_rewards(float): Standard deviation of rewards
        """
        
        # step for every theta (direction)
        step = np.zeros(self.theta.shape)
        n_best_deltas = len(rollouts)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += lr / (n_best_deltas * sigma_rewards) * step

class Model():
    """Training Model for agent."""
    def __init__(self):
        pass

    def make_env(self, env, video_freq, monitor_dir):
        """Set Gym Environment
        
        Args:
            env(str): OpenAI Gym environment name.
            video_freq(int): How often (step frequency) to record video.
            monitor_dir(str): Directory to write log and video files to.
        """

        self.env = gym.make(env)
        self.video_freq = video_freq
        should_record = lambda i: self.record_video
        self.record_video = False

        self.env = wrappers.Monitor(self.env, monitor_dir, 
            video_callable=should_record, force=True)

    def set_normalizer(self, normalizer):
        """Set Normalizer class to use for input normalizing."""
        self.normalizer = normalizer

    # Explore the policy on one specific direction and over one episode
    def explore(self, noise, direction=None, delta=None):
        """
        Run episode as exploration.

        Args:
            noise(float): Amount of delta distortion to add to weights.
            direction(str in ('-', '+') or None): Direction of next exploration.
            delta(list of floats): Relative step size (multiplied with noise) during gradient descent.

        Returns:
            Sum of rewards collected during episode.
        """
        state = self.env.reset()
        done = False
        num_plays = 0
        sum_rewards = 0.0
        while not done and num_plays < self.episode_length:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state, noise, delta, direction)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def train(self, n_steps, n_deltas, n_best_deltas, noise, episode_length, lr):
        """
        Train agent.

        Args:
            n_steps(int): Number of episodes.
            n_deltas(int): Number of deltas to use per episode.
            n_best_deltas(int): Number of best deltas to use for policy update.
            noise(float): Amount of delta distortion to add to weights.
            episode_length(int): Maximum number of actions to perform during an episode.
            lr(float): Learning rate of model.
        """
        self.episode_length = episode_length
        self.noise = noise
        for step in range(n_steps):
            # initialize the random noise deltas and the positive/negative rewards
            deltas = self.policy.sample_deltas(n_deltas=n_deltas)
            positive_rewards = [0] * n_deltas
            negative_rewards = [0] * n_deltas

            # play an episode each with positive deltas and negative deltas, collect rewards
            for k in range(n_deltas):
                positive_rewards[k] = self.explore(noise, direction="+", delta=deltas[k])
                negative_rewards[k] = self.explore(noise, direction="-", delta=deltas[k])
                
            # Compute the standard deviation of all rewards
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
            scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:n_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Update the policy
            self.policy.update(rollouts, lr, sigma_rewards)

            # Only record video during evaluation, every n steps
            if step % self.video_freq == 0:
                self.record_video = True
            # Play an episode with the new weights and print the score
            reward_evaluation = self.explore(noise)
            print('Step: ', step, 'Reward: ', reward_evaluation)
            self.record_video = False

if __name__ == '__main__':
    # openAI gym environment
    ENV_NAME = 'BipedalWalker-v2'
    
    # video directories
    video_dir = '../additional/videos'
    monitor_dir = os.path.join(video_dir, ENV_NAME)

    # create video directories
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)

    # hyperparameters for training
    hyperparams = {
        'n_steps': 1000, 
        'episode_length': 2000, 
        'lr': 0.01, 
        'n_deltas': 16, 
        'n_best_deltas': 16, 
        'noise': 0.03, 
        'video_freq': 50
    }

    # init model
    model = Model()

    # set environment and record params
    model.make_env(ENV_NAME, 
        video_freq=hyperparams['video_freq'], 
        monitor_dir=monitor_dir)
    
    # define input and output layer sizes
    input_size = model.env.observation_space.shape[0]
    output_size=model.env.action_space.shape[0]

    # initialize policy to use for training
    model.policy = Policy(input_size=input_size, output_size=output_size)

    # specify normalizing method to use
    model.set_normalizer(Normalizer(n_inputs=input_size))

    # train model
    model.train(n_steps=hyperparams['n_steps'], 
        n_deltas=hyperparams['n_deltas'], 
        n_best_deltas=hyperparams['n_best_deltas'], 
        noise=hyperparams['noise'], 
        episode_length=hyperparams['episode_length'], 
        lr=hyperparams['lr'])

    