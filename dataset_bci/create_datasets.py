import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mne
import scipy.io
import numpy as np
from data import TrialEEG, DatasetEEG

event_codes = {
    276 : 'Idle Eyes Open',
    277 : 'Idle Eyes Closed',
    768 : 'Trial Start',
    769 : 'Cue 1',
    770 : 'Cue 2',
    771 : 'Cue 3',
    772 : 'Cue 4',
    783 : 'Cue unkwnown',
    1023 : 'Rejected trial',
    1072 : 'Eye movements',
    32766 : 'Start of a new run'
    }

for n in range(1, 10):
    for mode in ['T','E']:
        
        filename = f'A0{n}{mode}'
        data = mne.io.read_raw_gdf(f'dataset_bci/original/BCICIV_2a_gdf/{filename}.gdf', verbose=False)

        # Informazioni sul file considerato
        subject_number = filename[1:3]
        if filename[3] == 'T': dataset_type = 'Training'
        if filename[3] == 'E': dataset_type = 'Test'

        # Voglio solo i trial senza artefatti?
        only_good_trials = True
        if dataset_type == 'Test': only_good_trials = False

        # Se è un file di test, carico anche le label
        if dataset_type == 'Test':
            test_labels = scipy.io.loadmat(f'dataset_bci/original/test_labels/{filename}.mat')['classlabel']

        # Estraggo la matrice dei dati
        data_matrix = data.get_data()
        times = data.times 

        # Estraggo gli eventi dalle annotazioni
        events, events_id_map = mne.events_from_annotations(data, verbose=False)
        events_id_map_reverse = {events_id_map[key]:int(key) for key in events_id_map}
        events_string = [event_codes[events_id_map_reverse[x]] for x in list(events[:,2])]

        labels_dict = {f'Cue {i+1}': i for i in range(4)}

        trials_data = []
        trials_times = []
        trials_labels = []
        good_trials = []

        # Ciclo sugli eventi 
        for i in range(events.shape[0]-1):

            event_time = events[i,0]
            event_id = events_string[i]

            if event_id == 'Trial Start':

                shift = 1
                next_event_id = events_string[i+shift]
                next_event_time = events[i+shift,0]

                # Controllo se è un trial rigettato
                if next_event_id == 'Rejected trial':
                    shift = 2
                    next_event_id = events_string[i+shift]
                    next_event_time = events[i+shift,0]
                    good_trial = False
                else:
                    good_trial = True

                
                # La label è proprio l'evento nel caso del training, mentre va presa a parte per il test
                if dataset_type == 'Training':
                    label = labels_dict[next_event_id]
                else:
                    label = test_labels[len(trials_data),0] - 1

                # Devo dedurre il tempo di fine del trial vedendo il successivo
                if i+shift+1 < events.shape[0]:
                    next_trial_time = events[i+shift+1,0] - 1
                else:
                    # Sono finiti i trial, quindi prendo l'ultimo punto
                    next_trial_time = data_matrix.shape[1]
                
                # Extract the trail info
                trial_data = data_matrix[0:-3,event_time:next_trial_time]
                trial_times = times[event_time:next_trial_time]

                # Shift times so that the cue is a t = 0
                trial_times = trial_times - times[next_event_time]

                # Conto il numero di punti fuori scala
                bad_points_num = np.sum(trial_data > 9.9e-5)
                if bad_points_num >= 10: good_trial = False
                good_trials.append(good_trial)

                trials_data.append(trial_data)
                trials_times.append(trial_times)
                trials_labels.append(label)

        
        # Costruisco i trial nel mio formato dati
        trials = []

        for i in range(len(trials_data)):
            trial = TrialEEG(trials_data[i], trials_labels[i], trials_times[i])
            trials.append(trial)

        info = {'subject': n, 'fs': 250}
        dataset = DatasetEEG(trials, info)

        save_filename = f'dataset_bci/processed/include_rejected/{subject_number}_{dataset_type}.dataset'
        dataset.save(save_filename)

        # Salvo un dataset con i soli trial buoni
        trials_good = [trials[i] for i in range(len(trials)) if good_trials[i]]

        info = {'subject': n, 'fs': 250}
        dataset_good = DatasetEEG(trials_good, info)
        save_filename = f'dataset_bci/processed/exclude_rejected/{subject_number}_{dataset_type}.dataset'
        dataset_good.save(save_filename)

