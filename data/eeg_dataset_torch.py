import torch
from data import DatasetEEG
from torch.utils.data import Dataset
from helpers.eeg_utils import convert_labels_string_to_int


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
