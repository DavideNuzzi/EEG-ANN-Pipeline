import numpy as np

def get_reactivation_strenth_matrix(states, tau):
    
    x = states

    x_shifted = np.roll(x, -tau, axis=0)

    # Taglio gli estremi
    x = x[0:-tau-1,:]
    x_shifted = x_shifted[0:-tau-1,:]

    # Risolvo il modello lineare
    beta = np.linalg.pinv(x.T @ x) @ x.T @ x_shifted
    
    return beta

def get_sequenceness(beta, transition_matrices=None):

    num_states = beta.shape[0]

    if transition_matrices is None:
        T_F = np.triu(np.ones((num_states, num_states)), k=1) - np.triu(np.ones((num_states, num_states)), k=2)
        T_B = T_F.T
        T_auto = np.eye(num_states, num_states)
        T_const = np.ones((num_states, num_states))
        transition_matrices = [T_F, T_B, T_auto, T_const]
    else:
        if type(transition_matrices) is tuple:
            transition_matrices = list(transition_matrices)

        if type(transition_matrices) is not list:
            transition_matrices = [transition_matrices]

    # Ora risolvo il problema al secondo ordine
    B_vec = np.reshape(beta, (-1,1))

    transition_matrices_reshaped = [np.reshape(M, (-1, 1)) for M in transition_matrices]
    A = np.concatenate(transition_matrices_reshaped, axis=1)
    z = np.linalg.pinv(A.T @ A) @ A.T @ B_vec

    return z[:,0]

def permutation_analysis(beta, transition_matrices, samples=250):

    if type(transition_matrices) is tuple:
        transition_matrices = list(transition_matrices)

    if type(transition_matrices) is not list:
        transition_matrices = [transition_matrices]

    num_states = beta.shape[0]
    num_transitions = len(transition_matrices)

    sequenceness = np.zeros((samples,num_transitions))

    for n in range(samples):

        # Permuto scambiando righe e colonne contemporaneamente
        uniform_idx = np.arange(num_states)
        rand_idx = uniform_idx
        while np.all(uniform_idx == rand_idx):
            rand_idx = np.random.permutation(num_states)
        
        T_F = transition_matrices[0]
        T_F = T_F[rand_idx, :][:, rand_idx]
        T_B = T_F.T

        transition_matrices[0] = T_F
        transition_matrices[1] = T_B

        # Calcolo la sequenceness
        z = get_sequenceness(beta, transition_matrices)
        sequenceness[n, :] = z

        # beta_shuffled = beta[rand_idx, :][:, rand_idx]

        # # Calcolo la sequenceness
        # z = get_sequenceness(beta_shuffled, transition_matrices)
        # sequenceness[n, :] = z

    return sequenceness