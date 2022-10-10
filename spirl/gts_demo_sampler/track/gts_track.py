
import pandas as pd
from spirl.utils.general_utils import AttrDict
class GTSTrack:
    def __init__(self, X=None, Y=None, Psi=None, Theta=None, Phi=None, Wl=None, Wr=None, Kap=None,
                 Left_X=None, Left_Y=None, Right_X=None, Right_Y=None, s=None,
                 track_type=None):
        self.suffix = track_type

        self.X = X
        self.Y = Y
        self.Psi = Psi
        self.Theta = Theta
        self.Phi = Phi
        self.Wl = Wl
        self.Wr = Wr
        self.Kap = Kap

        self.Left_X = Left_X
        self.Left_Y = Left_Y
        self.Right_X = Right_X
        self.Right_Y = Right_Y
        self.s = s

    def load_from_csv(self, track_dir):
        data = pd.read_csv(track_dir)
        track = AttrDict(data)
        return track
