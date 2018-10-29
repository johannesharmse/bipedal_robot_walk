import os

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

    