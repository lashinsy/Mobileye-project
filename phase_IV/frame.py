class Frame:

    def __init__(self, frame_path, frame_iteration):
        self.path = frame_path
        self.iteration = frame_iteration
        self.tfl_candidates = []
        self.tfl_auxiliary = []
