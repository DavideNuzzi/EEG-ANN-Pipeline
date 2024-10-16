from .classifiers import ShallowNet, DeepNet, EEGNet, EEGNetImproved, EEGNetMultiscale
from .vae import VAE, VAEClassifier
from .recurrent import LSTMClassifier, LSTMClassifierAllTimes, LSTMClassifierTimeMask, LSTMClassifierAttention
from .contrastive import EncoderContrastiveWeights, EncoderInfoNCE

all = [
    'ShallowNet',
    'DeepNet',
    'EEGNet',
    'EEGNetImproved',
    'EEGNetMultiscale',
    'VAE',
    'VAEClassifier',
    'EncoderInfoNCE',
    'EncoderContrastiveWeights'
]
