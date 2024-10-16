import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def count_model_parameters(model):
    model_pars_num = 0

    for p in model.parameters():
        s = p.size()
        n = 1
        for ps in s:
            n *= ps
        model_pars_num += n
    return model_pars_num


def model_summary(model):
        
    print(f'{"name":<50}:  count\n')
    for name, p in model.named_parameters():
        s = p.size()
        n = 1
        for ps in s:
            n *= ps

        print(f'{name:<50}:  {n}')


def plot_training_metrics(metrics):

    metric_names = list(dict.fromkeys([key.replace('_training','').replace('_validation','') for key in metrics]))
    
    cols = 2
    rows = int(np.ceil(len(metric_names) / 2))
    if rows == 0: rows = 1

    plt.figure(figsize=(10,rows*5))

    for i, metric_name in enumerate(metric_names):

        plt.subplot(rows, cols, i + 1)

        x = metrics[metric_name + '_training']
        plt.plot(x)

        if metric_name + '_validation' in metrics:
            x = metrics[metric_name + '_validation']
            plt.plot(x)
            plt.legend(['Training','Validation'])
        
        plt.xlabel('Epoch')
        plt.title(metric_name.replace('_history',''))
    plt.show()
    

def train_model(model, optimizer, dataloader_training, dataloader_validation=None, epochs=100,
                early_stopping=False, patience=10, stopping_metric='loss'):

        # Inizializzo una lista in cui inserire tutte le metriche per ogni epoca
        metrics_history = []

        # Se devo fare early stopping ho bisogno di salvare il modello migliore
        if early_stopping:
            if stopping_metric == 'accuracy':
                stopping_metric_best = -1e10
            else:
                stopping_metric_best = 1e10
            patience_epoch = 0

        # Ciclo sulle epoche mostrando una progressbar
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            
            # Inizializzo un dizionario per le metriche di questa epoca
            metrics_epoch_training = dict()

            # ------------------------------ Training Phase ------------------------------ #

            # Modalità training (per batchnorm e dropout)
            model.train()

            # Ciclo sui batch
            for i, batch in enumerate(dataloader_training):
                
                # Eseguo uno step di training
                batch_metrics = model.process_batch(batch, optimizer, is_eval=False)

                # Accumulo le metriche per questa epoca
                for metric in batch_metrics:
                    if metric not in metrics_epoch_training:
                        metrics_epoch_training[metric] = 0
                    metrics_epoch_training[metric] += batch_metrics[metric]

            # Medio le metriche
            for metric in metrics_epoch_training:
                metrics_epoch_training[metric] /= len(dataloader_training) * dataloader_training.batch_size

            # -------------------------------- Test Phase -------------------------------- #
            if dataloader_validation is not None:

                metrics_epoch_validation = dict()

                # Modalità eval (per batchnorm e dropout)
                model.eval()

                # Ciclo sui batch
                for i, batch in enumerate(dataloader_validation):
                    
                    # Eseguo uno step di test
                    batch_metrics =  model.process_batch(batch, None, is_eval=True)

                    # Accumulo le metriche per questa epoca
                    for metric in batch_metrics:
                        if metric not in metrics_epoch_validation:
                            metrics_epoch_validation[metric] = 0
                        metrics_epoch_validation[metric] += batch_metrics[metric]

                # Medio le metriche
                for metric in metrics_epoch_validation:
                    metrics_epoch_validation[metric] /= len(dataloader_validation) * dataloader_validation.batch_size

            # Mostro i risultati (training o validation, a seconda del caso) nella progressbar
            if dataloader_validation is None:
                pbar.set_postfix(metrics_epoch_training)
            else:
                pbar.set_postfix(metrics_epoch_validation)

            # Unisco le metriche di training e validation se necessario e le salvo nella lista che ne salva la storia
            metrics_epoch = dict()
            for metric in metrics_epoch_training:
                metrics_epoch[metric + '_training'] = metrics_epoch_training[metric]
            
            if dataloader_validation is not None:
                for metric in metrics_epoch_validation:
                    metrics_epoch[metric + '_validation'] = metrics_epoch_validation[metric]

            metrics_history.append(metrics_epoch)

            # Controllo se devo fermarmi per early stopping
            if early_stopping:
                if dataloader_validation is not None:
                    stopping_metric_value = metrics_epoch_validation[stopping_metric]

                    improved = False
                    if stopping_metric == 'accuracy':
                        if stopping_metric_value > stopping_metric_best:
                            improved = True
                    else:
                        if stopping_metric_value < stopping_metric_best:
                            improved = True

                    if improved:
                        stopping_metric_best = stopping_metric_value
                        best_model_state = model.state_dict()
                        patience_epoch = 0
                    else:
                        if patience_epoch > patience:
                            model.load_state_dict(best_model_state)
                            break
                        else:
                            patience_epoch += 1

        # Costrusico un dizionario finale in cui i risultati per ogni epoca sono messi tutti in una lista
        metrics_history_dict = dict()
        for i, m_epoch in enumerate(metrics_history):
            if i == 0:
                for m in m_epoch:
                    metrics_history_dict[m] = []
        
            for m in m_epoch:
                metrics_history_dict[m].append(m_epoch[m])

        return metrics_history_dict
