import numpy as np
import pickle as pkl
import torch
from typing import List
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from utils import check_number_timepoints, check_labels_type, convert_labels_string_to_int
import random
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader


# Questa è la classe più generale con cui descrivere un singolo trial di dati.
# Contiene l'attività EEG, la label del trial e i tempi. Se necessario, può
# essere ampliata in modo da contenere ulteriori informazioni.
class TrialEEG:

    def __init__(self, eeg_signals, label, timepoints):

        self.eeg_signals = np.array(eeg_signals, dtype=np.float32)
        self.label = label
        self.num_channels, self.num_timepoints = self.eeg_signals.shape
        self.timepoints = np.array(timepoints, dtype=np.float32)

    def __str__(self):
        info_string = f'Numero canali = {self.num_channels}, numero timepoints = {self.num_timepoints}\n'
        if self.timepoints is not None:
            info_string += f'Istante iniziale = {self.timepoints[0]}, tempo finale = {self.timepoints[-1]}'
        return info_string

    def plot(self, split_channels=True):

        if split_channels:

            # Mostro i segnali uno sotto l'altro
            # La separazione è pari a 4 deviazioni standard in modo che non si sovrappongano
            std = np.std(self.eeg_signals)
            tick_pos = []
            tick_labels = []

            for i in range(self.num_channels):

                y_shift = 4 * std * i
        
                plt.plot(self.timepoints, self.eeg_signals[i,:] + y_shift, color='k', linewidth=0.5)
                
                tick_pos.append(y_shift)
                tick_labels.append(str(i+1))

            plt.yticks(tick_pos, tick_labels)
            plt.ylabel('Channel')

        else:

            # Se invece devo mostrarli tutti assieme, uso un color coding
            # per differenziale i vari canali
            cmap = cm.get_cmap('jet') 
            for i in range(self.num_channels):
                color = cmap(i/(self.num_channels - 1))
                plt.plot(self.timepoints, self.eeg_signals[i,:], color=color, linewidth=0.5, alpha=0.5)

        # Alcune cose non dipendono dal tipo di plot
        plt.xlabel('Time')


class DatasetEEG():

    def __init__(self, trials: List[TrialEEG], info=None):

        self.trials = trials

        # Dizionario con le info sul dataset facoltative
        self.info = info

        # Altre info necessarie
        self.num_trials = len(trials)
        self.num_channels = trials[0].eeg_signals.shape[0]
        self.num_timepoints = check_number_timepoints(trials)
        self.labels_type, self.labels_format = check_labels_type(trials)

    
    # Funzione per salvare il dataset su file (per ora con pickle, in futuro con altri formati)
    def save(self, filepath):
        with open(filepath,'wb') as f:
            pkl.dump(self, f)

    # Funzione per caricare (statica perché prima di caricare il dataset non esiste)
    @staticmethod
    def load(filepath):
        with open(filepath,'rb') as f:
            dataset = pkl.load(f)
        return dataset

    # Mostro le info del dataset quando chiamo "print" su di esso
    def __str__(self):
        info_string = f"{'num_trials':25}:  {self.num_trials}\n"
        info_string += f"{'num_channels':25}:  {self.num_channels}\n"
        info_string += f"{'num_timepoints':25}:  {self.num_timepoints}\n"
        info_string += f"{'labels_type':25}:  {self.labels_type}\n"
        info_string += f"{'labels_format':25}:  {self.labels_format}\n"

        if self.info:
            for key in self.info:
                info_string += f'{key:<25}:  {self.info[key]}\n'
        return info_string

    # Estrae un subset di trial e crea un nuovo dataset a partire da questi
    def extract_subset(self, idx):

        trials_subset = [self.trials[i] for i in range(self.num_trials) if i in idx]
        return DatasetEEG(trials_subset, self.info)
    
    # Splitta il dataset in training e validation
    def split_dataset(self, validation_size=0.2):

        # Quanti trial prendere
        split_idx = int(np.round((1-validation_size) * self.num_trials))

        # Li prendo a caso e non sequenzialmente
        random_indices = list(np.random.permutation(self.num_trials))
        train_indices = random_indices[0:split_idx]
        validation_indices = random_indices[split_idx:]

        # Creo i due dataset
        dataset_train = self.extract_subset(train_indices)
        dataset_validation = self.extract_subset(validation_indices)

        return dataset_train, dataset_validation
    
class DatasetEEGTorch(Dataset):

    def __init__(self, dataset: DatasetEEG):
        
        # Controllo che i trials siano tutti della stessa lunghezza
        # altrimenti non posso creare il dataset pytorch
        if not isinstance(dataset.num_timepoints, int):
            raise ValueError('I trial non hanno tutti la stessa lunghezza')
        
        # Mantengo un riferimento al dataset da cui previene
        # e altre informazioni utili
        self.dataset_original = dataset
        self.labels_type = dataset.labels_type
        self.labels_format = dataset.labels_format

        # Per comodità salvo anche qui come attributi le info sul dataset
        self.num_trials = dataset.num_trials
        self.num_timepoints = dataset.num_timepoints
        self.num_channels = dataset.num_channels
        
        # Creo un tensore per tenere tutti i dati
        self.eeg_signals = torch.zeros((self.num_trials, 1, self.num_channels, self.num_timepoints), dtype=torch.float32)

        for i, trial in enumerate(dataset.trials):
            self.eeg_signals[i, 0, :, :] = torch.from_numpy(trial.eeg_signals)

        # Caso singola label: creo un tensore  (se sono stringhe devo prima convertirle in interi)
        # Caso multi-label: creo un dizionario e a ogni tipo di label associo un tensore
        self.create_labels()

    def __len__(self):
        return self.num_trials

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eeg_signals[idx, :, :, :]

        # Le label dipendono dal caso
        if self.labels_type == 'single_label':
            y = self.labels[idx]
        else:
            y = {label_name: self.labels[label_name][idx] for label_name in self.label_names }

        return x, y
    
    def create_labels(self):

        # Separo il caso single-label da quello multi-label

        # Caso single label
        if self.labels_type == 'single_label':

            # Creo una lista con le label
            labels = [trial.label for trial in self.dataset_original.trials]

            # Controllo se sono stringhe
            if self.labels_format == 'string':

                # Nel caso le converto e salvo il dizionario per tornare indietro
                labels, labels_int_to_str = convert_labels_string_to_int(labels)
                self.labels_int_to_str = labels_int_to_str

            # Trasformo le label in un tensore e le salvo
            self.labels = torch.LongTensor(labels)
            
        else:

            # Nel caso multilabel produco un tensore per ogni tipo di variabile
            # e poi li metto in un dizionario
            self.label_names = list(self.dataset_original.trials[0].label.keys())

            # Dizionario in cui mettere le label convertite
            labels = dict()

            # Dizionario in cui mettere i vari dizionari per la conversione da int a str (se multipli)
            labels_int_to_str = dict()
            
            # Ciclo su ogni tipo di label nella lista
            for label_name in self.label_names:

                # Creo una lista con le label relative a questa chiave
                labels_i = [trial.label[label_name] for trial in self.dataset_original.trials]
                
                # Controllo se sono stringhe e nel caso le converto
                if self.labels_format[label_name] == 'string':

                    labels_i, labels_int_to_str_i = convert_labels_string_to_int(labels_i)
                    labels_int_to_str[label_name] = labels_int_to_str_i

                # Converto in base anche al tipo di dato
                if self.labels_format[label_name] == 'float':
                    labels_i = torch.FloatTensor(labels_i)
                else:
                    labels_i = torch.LongTensor(labels_i)

                # Le salvo nel dizionario globale
                labels[label_name] = labels_i
            
            # Salvo le label nella classe
            self.labels = labels

            # Salvo il dizionario di conversione se non è vuoto
            if len(labels_int_to_str) > 0:
                self.labels_int_to_str = labels_int_to_str            

    def to_device(self, device):

        # Sposto i dati sul device
        self.eeg_signals = self.eeg_signals.to(device)

        # Sposto le label sul device, tenendo conto dei possibili casi
        if self.labels_type == 'single_label':
            self.labels = self.labels.to(device)
        else:
            for key in self.labels:
                self.labels[key] = self.labels[key].to(device)



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
        

