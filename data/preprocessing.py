import numpy as np
from data import TrialEEG, DatasetEEG
from helpers.eeg_utils import check_number_timepoints

# Normalizza i segnali nel dataset
# Assumo che le uniche normalizzazioni possibili siano lungo i tempi
# oppure lungo i tempi e i trial contemporaneamente.
# Il parametro "across_trials" controlla questo comportamento.
def normalize_signals(dataset: DatasetEEG, across_trials=True):

    # Trials del dataset
    # Poiché la classe viene passata per riferimento, la normalizzazione
    # modificherà direttamente i trials, senza il bisogno di creare un nuovo dataset
    trials = dataset.trials

    # Controllo se devo normalizzare separatamente per ogni trial
    if across_trials:

        # Se devo mettere assieme i trial li concateno prima lungo i tempi
        signals_concat = np.zeros((dataset.num_channels, 0))

        for trial in trials:
            signals_concat = np.concatenate((signals_concat, trial.eeg_signals), axis=1)

        # Calcolo media e std e poi normalizzo ogni trial separatamente
        mean = np.mean(signals_concat, axis=(0,1), keepdims=True)
        std = np.std(signals_concat, axis=(0,1), keepdims=True)
        for trial in trials:
            trial.eeg_signals = (trial.eeg_signals - mean) / std
    else:

        # Se devo normalizzare separatamente mi basta ciclare e farlo uno ad uno
        for trial in trials:
            x = trial.eeg_signals
            mean = np.mean(x, axis=1, keepdims=True)
            std = np.std(x, axis=1, keepdims=True)
            trial.eeg_signals = (x - mean) / std


# Normalizzo i trials rispetto a una baseline definita dall'utente
# (di solito l'intervallo di tempo prima dello stimolo/cue)
def normalize_to_baseline(dataset: DatasetEEG, t_min, t_max):
        
    trials = dataset.trials

    for trial in trials:
        x = trial.eeg_signals
        times = trial.timepoints

        # Identifico i punti tra t_min e t_max
        mask = (times >= t_min) & (times < t_max)

        # Maschero e medio
        x_baseline = x[:, mask]
        mean = np.mean(x_baseline, axis=1, keepdims=True)
        std = np.std(x_baseline, axis=1, keepdims=True)
        trial.eeg_signals = (x - mean) / std


# Taglio i segnali in ogni trial in modo che siano tra due tempi fissati
# Come risultato i trials hanno tutti la stessa lunghezza
def crop_signals(dataset: DatasetEEG, t_min, t_max):

    trials = dataset.trials

    mask_length = None

    for i in range(len(trials)):

        trial = trials[i]

        # Crea maschera per timepoints validi
        times = trial.timepoints
        mask = (times >= t_min) & (times < t_max)  

        # Alla prima iterazione vedo la lunghezza della maschera e la uso
        # come riferimento per tutte le altre
        if i == 0: mask_length = np.sum(mask)

        # Controllo se la maschera ha la dimensione giusta, altrimenti 
        # questo è un trial non valido e lo elimino
        if np.sum(mask) != mask_length:
            trials.pop(i)
            i -= 1
            continue

        # Ora uso la maschera per tagliare sia i segnali che i tempi
        trial.eeg_signals = trial.eeg_signals[:, mask]
        trial.timepoints = trial.timepoints[mask]
        trial.num_timepoints = len(trial.timepoints)

    # Alla fine devo aggiornare le info del dataset in quanto ora
    # ha tutti i trial della stessa lunghezza
    dataset.num_timepoints = check_number_timepoints(trials)
    dataset.num_trials = len(dataset.trials)
   

def create_dataset_from_windows(dataset, time_start, time_end, time_window, fs, shift=10):
    
    # Lo shift è dato in punti, mentre tutti gli altri parametri
    # sono dati come tempi in secondi
    dt = 1/fs
    shift = dt * shift        

    trials_new = []

    # Ciclo sui trial
    for n, trial in enumerate(dataset.trials):

        signals = trial.eeg_signals
        label = trial.label
        times = trial.timepoints
        
        if len(times) == 0: continue

        # Ciclo sulle finestre
        t_start = time_start
        t_end = t_start + time_window

        while t_end < time_end:
            mask = (times >= t_start) & (times < t_end)
            x = signals[:, mask]
            t = times[mask]

            t_start += shift
            t_end += shift

            trial_new = TrialEEG(x, label, t)
            trials_new.append(trial_new)

    return DatasetEEG(trials_new, info=dataset.info)
