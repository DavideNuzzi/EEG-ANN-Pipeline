import torch
import numpy as np
from data import DatasetEEGTorch

# ----------------------------- Standard InfoNCE ----------------------------- #

# Permette di samplare indici per costruire i sample positivi
# a partire da una label discreta
class SamplerDiscrete():

    def __init__(self, labels):

        # Se non è un array di numpy, lo converto
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        self.labels = labels

        # Ordino
        self.sorted_idx = torch.from_numpy(np.argsort(self.labels))

        # Distribuzione cumulativa
        self.counts = np.bincount(self.labels)
        self.cdf = np.zeros((len(self.counts) + 1,))
        self.cdf[1:] = np.cumsum(self.counts)

    def sample(self, reference_labels):

        # Converto le label se necessario
        if torch.is_tensor(reference_labels):
            reference_labels = reference_labels.cpu().numpy()

        idx = np.random.uniform(0, 1, len(reference_labels))
        idx *= self.cdf[reference_labels + 1] - self.cdf[reference_labels]
        idx += self.cdf[reference_labels]
        idx = idx.astype(int)

        return self.sorted_idx[idx]
    

# Permette di samplare indici per costruire i sample positivi
# a partire da una label continua
class SamplerContinuousGaussian():

    def __init__(self, labels, std=1):

        # Se non è un array di numpy, lo converto
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        # Se la dimensione è uno, li espando
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=-1)

        self.labels = labels
        self.std = std

        # Precalcolo i quadrati per semplificare il conto dopo
        self.labels_squared = self.labels**2

    def sample(self, reference_labels):

        # Converto le label se necessario
        if torch.is_tensor(reference_labels):
            reference_labels = reference_labels.cpu().numpy()

        # Se la dimensione è uno, li espando
        if reference_labels.ndim == 1:
            reference_labels = np.expand_dims(reference_labels, axis=-1)   

        # Samplo da una gaussiana
        samples = np.random.randn(*reference_labels.shape) * self.std + reference_labels

        # So che "self.labels" ha dimensione (N_trials, 1)
        # e che "reference_values" ha dimensione (batch_size, 1)
        # Voglio la distanza tra ogni elemento di reference_values e ogni 
        # elemento di self.labels. Quindi sfruttando un po' di trucchetti
        # di broacasting posso calcolare (x - y)^2 = x^2 + y^2 - 2xy
        samples_squared = (samples**2).T

        product_term = np.einsum('ni,mi->nm', self.labels, reference_labels)
        distance_matrix = self.labels_squared + samples_squared - 2 * product_term

        # Data la matrice di distanze, cerco gli elementi che la minimizzano per ogni reference
        return np.argmin(distance_matrix, axis=0)


# Il contrastive batch sampler deve fornire tre tensori, x, y_pos e y_neg
# tutti della stessa dimensione batch_size x channels x timepoints
# Per farlo genera prima gli esempi negativi e le reference casualmente
# Poi deve avere un criterio con cui samplare i positivi
class DataLoaderContrastive():

    def __init__(self, dataset: DatasetEEGTorch, samplers, batch_size, batches_per_epoch=1):

        self.dataset = dataset
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.samplers = samplers

    def sample_positive(self, reference_idx):

        if self.dataset.labels_type == 'single_label':     
            reference_labels = self.dataset.labels[reference_idx]
            return self.samplers.sample(reference_labels)
        else:
            positive_samples = dict()
            for label_name in self.samplers:
                reference_labels = self.dataset.labels[label_name][reference_idx]
                positive_samples[label_name] = self.samplers[label_name].sample(reference_labels) 
            return positive_samples
        
    def __iter__(self):

        for _ in range(self.batches_per_epoch):

            # Genero i numeri casuali per reference e negative
            rand_idx = np.random.choice(self.dataset.num_trials, self.batch_size * 2)

            # Prendo gli indici di reference e negative
            reference_idx = rand_idx[0:self.batch_size]
            negative_idx = rand_idx[self.batch_size:]

            # Prendo i sample relativi
            x = self.dataset.eeg_signals[reference_idx,:,:,:]
            y_neg = self.dataset.eeg_signals[negative_idx,:,:,:]

            # Per le positive devo samplare adeguatamente
            # Se i positive_idx sono una lista (più label)
            # devo creare un tensore più grande per gli y_pos 
            positive_idx = self.sample_positive(reference_idx)

            if self.dataset.labels_type == 'single_label':     
                y_pos = self.dataset.eeg_signals[positive_idx,:,:,:]
                labels = self.dataset.labels[reference_idx]
            else:
                y_pos = {label_name: self.dataset.eeg_signals[positive_idx[label_name],:,:,:] for label_name in positive_idx}
                labels = {label_name: self.dataset.labels[label_name][reference_idx] for label_name in positive_idx}
            yield x, y_pos, y_neg, labels


    def __len__(self):
        return self.batches_per_epoch
    

# ---------------------------- My modified InfoNCE --------------------------- #
class LabelsDistance():

    def __init__(self, labels_distance_functions):

        if type(labels_distance_functions) is dict:
            self.multi_label = True
        else:
            self.multi_label = False

        self.labels_distance_functions = labels_distance_functions
    

    def get_weights(self, labels):
    
        def pairwise_tensor_function(f, x):
            x = x.view((-1,1))
            return f(x, x.T)
        
        if self.multi_label:
            weights = []

            for label_name in labels:
                weights.append( pairwise_tensor_function(
                                self.labels_distance_functions[label_name],
                                labels[label_name]))
                            
            return torch.stack(weights, dim=-1)
        else:
            batch_size = len(labels)
            return pairwise_tensor_function(self.labels_distance_functions, labels).view((batch_size,batch_size,1))
        

