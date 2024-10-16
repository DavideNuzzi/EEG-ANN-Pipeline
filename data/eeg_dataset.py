import numpy as np
import pickle as pkl
import matplotlib.cm as cm
from typing import List
from matplotlib import pyplot as plt
from helpers.eeg_utils import check_number_timepoints, check_labels_type

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
