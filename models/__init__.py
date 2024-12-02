from .classifiers import ShallowNet, DeepNet, EEGNet, EEGNetImproved, EEGNetMultiscale, EEGNetAdaptive
from .vae import VAE, VAEClassifier
from .recurrent import LSTMClassifier, LSTMClassifierAllTimes, LSTMClassifierTimeMask, LSTMClassifierAttention
from .contrastive import EncoderContrastiveWeights, EncoderInfoNCE

__all__ = [
    'ShallowNet',
    'DeepNet',
    'EEGNet',
    'EEGNetImproved',
    'EEGNetMultiscale',
    'VAE',
    'VAEClassifier',
    'EncoderInfoNCE',
    'EncoderContrastiveWeights',
    'EEGNetAdaptive',
    'LSTMClassifier',
    'LSTMClassifierAllTimes',
    'LSTMClassifierTimeMask',
    'LSTMClassifierAttention'
    ]
