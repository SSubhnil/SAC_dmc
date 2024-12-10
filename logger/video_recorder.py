import imageio
import numpy as np

class VideoRecorder:
    def __init__(self, dir_name, fps=30, enabled=True):
        self.dir_name = dir_name
        self.enabled = enabled
        self.frames = []
        self.fps = fps

    def init(self, enabled=True):
        self.enabled = enabled
        self.frames = []

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array')
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled and len(self.frames) > 0:
            path = f"{self.dir_name}/{file_name}"
            imageio.mimsave(path, self.frames, fps=self.fps)
