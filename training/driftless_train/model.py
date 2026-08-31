"""Velocity/vibration-filter model.

Small enough to run at 10Hz on-device after TFLite export. Architecture
(1D-CNN vs. small GRU over the IMU window) is an open choice — pick after
looking at what IO-VNBD's window statistics actually look like, not before.
"""

import torch.nn as nn


class VelocityNet(nn.Module):
    # TODO: define once the windowing scheme in dataset.py is settled.
    pass
