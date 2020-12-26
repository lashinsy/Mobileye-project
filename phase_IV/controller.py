import pickle
from phase_IV.TFL_manager import TFL_manager
from phase_IV.frame import Frame


class Controller:

    def __init__(self):
        self.tfl_manager = None

    def get_frame_list(self):
        with open("list.pls") as frame_file:
            frame_list = []
            for f in frame_file:
                frame_list.append(f[:-1])

        return frame_list[0], frame_list[1:]

    def init_tfl(self, pickle_file):
        with open(pickle_file, 'rb') as pklfile:
            self.tfl_manager = TFL_manager(pickle.load(pklfile, encoding='latin1'))

    def run(self):
        pickle_file, frame_list = self.get_frame_list()
        self.init_tfl(pickle_file)

        for i, f in enumerate(frame_list):
            self.tfl_manager.on_frame(Frame(f, i + 1))


def main():
    c = Controller()
    c.run()


if __name__ == '__main__':
    main()
