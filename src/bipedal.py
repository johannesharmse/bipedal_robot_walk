import gym
import os


class Agent():
    def __init__(self):
        self.env = gym.make()

    def env(self, env):
        """Define Gym Environment
        
        Args:
            env(str): OpenAI Gym environment name.
        """
        
        self.env = gym.make(ENV_NAME)
        

    def fit(self, n_steps, episode_length, lr, 
    n_deltas, n_best_deltas, noise, 
    seed, env_name, log_freq):
    """Fit hyperparameters for agent training.

    Args:
        n_steps (int): Number of steps per 



    """




if __name__ == '__main__':
    # openAI gym environment
    ENV_NAME = 'BipedalWalker-v2'
    # video directories
    video_dir = './additional/videos'
    monitor_dir = video_dir + ENV_NAME

    # create video directories
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    if not os.path.exists(monitor_dir):
        os.makedirs(video_dir)

    