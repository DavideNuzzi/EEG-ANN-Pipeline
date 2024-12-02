from .shallow_net import ShallowNet
from .deep_net import DeepNet
from .EEGNet import EEGNet
from .eeg_net_improved import EEGNetImproved
from .eeg_net_multiscale import EEGNetMultiscale
from .eeg_net_adaptive import EEGNetAdaptive

__all__ = [
    'ShallowNet',
    'DeepNet',
    'EEGNet',
    'EEGNetImproved',
    'EEGNetMultiscale',
    'EEGNetAdaptive'
]
