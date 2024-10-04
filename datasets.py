import numpy as np
import pickle as pkl
import torch
from typing import List
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from utils import check_number_timepoints, check_labels_type
import random



# Questa è la classe più generale con cui descrivere un singolo trial di dati.
# Contiene l'attività EEG, la label del trial e i tempi. Se necessario, può
# essere ampliata in modo da contenere ulteriori informazioni.
class TrialEEG:

    def __init__(self, eeg_signals, label, timepoints):

        self.eeg_signals = np.array(eeg_signals, dtype=np.float32)
        self.label = label
        self.num_channels, self.num_timepoints = self.eeg_signals.shape
        self.timepoints = np.array(timepoints, dtype=np.float32)

    def plot_trial(self):
        pass

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
        self.labels_type = check_labels_type(trials)

    
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
    
    def create_pytorch_dataset(self, validation_size=0.2, standardization_method='crop', standardization_params=None):

        if validation_size > 0:

            split_idx = int(np.round((1-validation_size) * self.info.num_trials))
            
            random_indices = list(np.random.permutation(self.info.num_trials))
            train_indices = random_indices[0:split_idx]
            validation_indices = random_indices[split_idx:]
        
            if split_idx < self.info.num_trials - 1:
                trials_training = [self.trials[i] for i in train_indices]
                trials_validation = [self.trials[i] for i in validation_indices]

                dataset_training = DatasetEEGTorch(trials_training, standardization_method, standardization_params)
                dataset_validation = DatasetEEGTorch(trials_validation, standardization_method, standardization_params)
                return dataset_training, dataset_validation

        dataset_training = DatasetEEGTorch(self.trials, standardization_method, standardization_params)
        return dataset_training

    def plot_trials(self, idx):
        pass


class DatasetEEGTorch(Dataset):

    def __init__(self, dataset: DatasetEEG):
        
        # Controllo che i trials siano tutti della stessa lunghezza
        # altrimenti non posso creare il dataset pytorch
        if not isinstance(dataset.num_timepoints, int):
            raise ValueError('I trial non hanno tutti la stessa lunghezza')
        
        # Mantengo un riferimento al dataset da cui previene
        self.dataset_original = dataset

        # Per comodità salvo anche qui come attributi le info sul dataset
        self.num_trials = dataset.num_trials
        self.num_timepoints = dataset.num_timepoints
        self.num_channels = dataset.num_channels
        
        # Creo un tensore per tenere tutti i dati
        self.eeg_signals = torch.zeros((self.num_trials, 1, self.num_channels, self.num_timepoints), dtype=torch.float32)

        for i, trial in enumerate(dataset.trials):
            self.eeg_signals[i, 0, :, :] = torch.from_numpy(trial.eeg_signals)

        # Creo un tensore con le label (se sono stringhe devo prima convertirle in interi)
        self.create_labels_tensor()

    def __len__(self):
        return self.num_trials

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eeg_signals[idx, :, :, :]
        y = self.labels[idx]

        return x, y
    
    def create_labels_tensor(self):

        # Inizializzo il tensore
        self.labels = torch.zeros(self.num_trials, dtype=torch.long)

        # Se le labels sono stringhe, mi servono dei dizionari per convertirle
        if self.dataset_original.labels_type == 'string':
            self.labels_int_to_str = dict()
            self.labels_str_to_int = dict()

        for i, trial in enumerate(self.dataset_original.trials):
            
            label = trial.label

            # Se è una stringa la converto
            if self.dataset_original.labels_type == 'string':

                # Controllo se ho già trovato questa label
                if label not in self.labels_str_to_int:

                    # Aggiungo un elemento a entrambi i dizionari di conversione
                    new_label_int = len(self.labels_int_to_str)
                    self.labels_str_to_int[label] = new_label_int
                    self.labels_int_to_str[new_label_int] = label

                # La converto
                label = self.labels_str_to_int[label]
            
            self.labels[i] = trial.label

    def to_device(self, device):
        self.eeg_signals = self.eeg_signals.to(device)
        self.labels = self.labels.to(device)


# Questo dataset sfrutta l'idea di CEBRA, ma la vicinanza
# tra due sample "x" è data dalla label stessa "y".
# Quindi quando genero un batch di dati, devo generare:
# x     :   (batch_size, num_channels, num_times)
# y_p   :   (batch_size, num_channels, num_times)
# y_m   :   (batch_size, 10, num_channels, num_times)
# y_p e y_m vengono samplati uniformemente tra tutti i trial con la stessa label
class DatasetEEGTorchCebra(DatasetEEGTorch):
    
    def __init__(self, dataset: DatasetEEG):
        
        # Inizializzo la classe madre
        super().__init__(dataset)
        # DatasetEEGTorch.__init__(self, dataset)

    def __getitem__(self, idx):

        # Prendo un sample
        x = self.eeg_signals[idx, :, :, :]

        # Labels
        label = self.labels[idx]

        # Trovo un esempio positivo e dieci negativi
        y_pos = self.get_sample_from_label(label, exclude_ind=idx)
        y_neg = self.get_negative_samples(label, device=x.device, num=10)

        return x, y_pos, y_neg, label
    
    
    # Restituisce un sample che ha la label scelta
    # Non può restituire il sample che ha lo stesso id di "exclude_ind"
    # Se complementary = True, restituisce il sample con la label indicata
    # Se complementary = False, restituisce un sample con una label diversa da quella indicata
    def get_sample_from_label(self, label, exclude_ind, complementary=False):

        # Ripeto fino a quando non ho trovato
        found = False

        while not found:

            # Scelgo a caso un indice
            random_ind = random.randint(0, self.num_trials - 1)

            # Escludo nel caso sia l'indice vietato
            if random_ind == exclude_ind: continue

            # Prendo la label corrispondente
            sample_label = self.labels[random_ind]

            if complementary is False:
                if sample_label == label:
                    return self.eeg_signals[random_ind,:,:,:]
            else:
                if sample_label != label:
                    return self.eeg_signals[random_ind,:,:,:]
    
    # Funzione speciale che restituisce più negative samples alla volta
    # Non serve un argomento "exclude_ind" perché per i negative samples
    # non può mai accadere che coincidano con l'indice di x
    def get_negative_samples(self, label, device, num=10,):
        
        y_neg = torch.zeros((num, 1, self.num_channels, self.num_timepoints), device=device)
        
        for i in range(num):
            y_neg[i,:,:,:] = self.get_sample_from_label(label, exclude_ind=-1, complementary=True)

        return y_neg


 