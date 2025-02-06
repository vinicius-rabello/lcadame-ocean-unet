import numpy as np
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator


#================================================================================
#================================================================================

def inverse_SMW(ensemble,sig_m,R,Nlat,Nlon):
    ensemble_mean = np.mean(ensemble,0)
    deviations = ensemble - ensemble_mean
    A = deviations.T  # Transpose to make deviations as columns
    # print(np.shape(A))
    AT = deviations
    N = ensemble.shape[0]
    PP = (A @ AT) / (N - 1)
    # Pb = (1/(N-1)) * (deviations.reshape(1,-1)).T @ (deviations.reshape(1,-1))
    ######  R Inverse  #########################################
    # rInv=np.linalg.inv(R)
    # Or
    rInv = (1/ sig_m ** 2) * np.eye(Nlat * Nlon * 2, Nlat * Nlon * 2)     # Cheaper
    ################################################################
    sInv= rInv - rInv @ A @ np.linalg.inv((N-1)*np.eye(N)+ AT @ rInv @ A ) @ AT @ rInv
    ###############################################
    return sInv






#================================================================================
#================================================================================




def approximate_inverse_cg(A, tol=1e-6, max_iter=1000):
    n = A.shape[0]
    I = np.eye(n)  # Identity matrix
    A_inv_approx = np.zeros_like(A)  # Placeholder for the inverse approximation

    def matvec_A(x):
        return A @ x

    A_op = LinearOperator((n, n), matvec=matvec_A)

    for i in range(n):
        e_i = I[:, i]
        x_i, info = cg(A_op, e_i, tol=tol, maxiter=max_iter)
        if info != 0:
            print(f"Warning: Conjugate Gradient did not converge for column {i}")
        A_inv_approx[:, i] = x_i

    return A_inv_approx


def convert_to_1d_index(index, shape):
    ind = index[0] * (shape[1] * shape[2]) + index[1] * shape[2] + index[2]
    # print(ind)
    return ind

#####################################################################################
# Ensemble Kalman filter
#####################################################################################



def EnKF4(ubi, observations, numericalState, H, N, M, Nlat, Nlon, loss_percentage):
    """
    Perform an EnKF update with a given percentage of observations randomly set to NaN.

    Parameters:
    ubi (numpy.ndarray): Ensemble of background states (N x state_dim).
    observations (numpy.ndarray): The observation matrix (2 x Nlat x Nlon).
    numericalState (numpy.ndarray): The numerical state (M x state_dim).
    H (numpy.ndarray): The observation matrix (obs_dim x state_dim).
    N (int): Number of ensemble members.
    M (int): Number of states.
    Nlat (int): Number of latitude points.
    Nlon (int): Number of longitude points.
    loss_percentage (float): The percentage of observations to ignore (0 to 1).

    Returns:
    numpy.ndarray: The updated state after data assimilation (M x state_dim).
    """

    # The analysis step for the (stochastic) ensemble Kalman filter with virtual observations
    obs = np.copy(observations)

    # Flatten the observations and identify the received (non-NaN) indices
    obs_flat = np.array(obs).flatten()
    total_indices = np.arange(len(obs_flat))
    received_indices = np.where(~np.isnan(obs_flat))[0]

    # Determine the number of observations to drop
    num_lost = int(len(received_indices) * loss_percentage)

    # Randomly select the indices to drop
    lost_indices = np.random.choice(received_indices, num_lost, replace=False)

    # Set the selected observations to NaN
    obs_flat[lost_indices] = np.nan

    # Reshape the observations back to the original shape
    obs = obs_flat.reshape(observations.shape)

    ub = np.mean(ubi, 0)
    Pb = (1 / (N - 1)) * (ubi - ub.reshape(1, -1)).T @ (ubi - ub.reshape(1, -1))

    # Initialize the H matrix
    H = np.zeros((2 * Nlat * Nlon, 2 * Nlat * Nlon))

    sig_m = 0.15
    R = sig_m ** 2 * np.eye(2 * Nlat * Nlon, 2 * Nlat * Nlon)

    # Flatten the observations and identify the received (non-NaN) indices again
    obs_flat = np.array(obs).flatten()
    received_indices = np.where(~np.isnan(obs_flat))[0]

    # Create a reduced observation matrix H_received for the remaining observations
    H_received = H[received_indices, :]

    # Reduce the R matrix to correspond to the remaining observations
    R_received = R[np.ix_(received_indices, received_indices)]

    # Create the remaining observations vector
    obs_remaining = obs_flat[received_indices] + np.random.normal(0, sig_m, size=len(received_indices))

    # Compute the Kalman gain using H_received
    D = H_received @ Pb @ H_received.T + R_received
    K = Pb @ H_received.T @ np.linalg.inv(D)

    # State after data assimilation
    stateAfterDA = np.zeros([M, Nlat * Nlon * 2])

    for i in range(M):
        # Innovation for the remaining observations
        innovation = obs_remaining - H_received @ numericalState[i, :]
        stateAfterDA[i, :] = numericalState[i, :] + K @ innovation

    return stateAfterDA


# # Example usage
# N = 100
# M = 50
# Nlat = 46
# Nlon = 68
# ubi = np.random.randn(N, Nlat * Nlon * 2)
# observations = np.random.randn(2, Nlat, Nlon)
# numericalState = np.random.randn(M, Nlat * Nlon * 2)
# H = np.random.randn(2 * Nlat * Nlon, Nlat * Nlon * 2)
# loss_percentage = 0.2  # For example, ignore 20% of observations
#
# updated_state = EnKF3(ubi, observations, numericalState, H, N, M, Nlat, Nlon, loss_percentage)
# print(updated_state)

