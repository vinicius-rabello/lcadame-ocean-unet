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
def EnKFHP(ubi, observations, numericalState, N, M, Nlat, Nlon,loss_percentage):
    # The analysis step for the (stochastic) ensemble Kalman filter
    # with virtual observations


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
    print('Inside KF, ub',np.shape(Pb),ub.shape)

    size = 2 * Nlat * Nlon
    H = np.eye(size)

    sig_m = 0.15
    R = sig_m ** 2 * np.eye(2 * Nlat * Nlon, 2 * Nlat * Nlon)
    ########################################################################
    # Shape of the 3D matrix
    shape = (46, 68, 2)

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
        # print( numericalState[i, :])

    return stateAfterDA


def gaspari_cohn(distances, radius):
    # Gaspari-Cohn localization function
    result = np.zeros_like(distances)

    # Apply localization within the given radius (5l in this case)
    for i, d in enumerate(distances):
        if d <= radius:
            if d <= radius / 2:
                result[i] = (1 - 5 * (d / radius) ** 2 + 4 * (d / radius) ** 3)
            else:
                result[i] = 4 * (1 - (d / radius)) ** 2 / (radius / d)

    return result


def EnKFHP_Loc(ubi, observations, numericalState, N, M, Nlat, Nlon, loss_percentage):
    # The analysis step for the (stochastic) ensemble Kalman filter with localization

    # Radius for localization (5l)
    # radius = 5 * l

    # Flatten the observations and identify the received (non-NaN) indices
    obs = np.copy(observations)
    obs_flat = np.array(obs).flatten()
    total_indices = np.arange(len(obs_flat))
    received_indices = np.where(~np.isnan(obs_flat))[0]

    # Determine the number of observations to drop
    num_lost = int(len(received_indices) * loss_percentage)
    lost_indices = np.random.choice(received_indices, num_lost, replace=False)
    obs_flat[lost_indices] = np.nan
    obs = obs_flat.reshape(observations.shape)

    # Compute background ensemble mean and covariance matrix Pb
    ub = np.mean(ubi, 0)
    Pb = (1 / (N - 1)) * (ubi - ub.reshape(1, -1)).T @ (ubi - ub.reshape(1, -1))

    size = 2 * Nlat * Nlon
    H = np.eye(size)

    sig_m = 0.15
    R = sig_m ** 2 * np.eye(size)

    # Flatten the observations and identify the received (non-NaN) indices again
    obs_flat = np.array(obs).flatten()
    received_indices = np.where(~np.isnan(obs_flat))[0]

    # Create a reduced observation matrix H_received for the remaining observations
    H_received = H[received_indices, :]

    # Reduce the R matrix to correspond to the remaining observations
    R_received = R[np.ix_(received_indices, received_indices)]

    # Remaining observations vector with noise
    obs_remaining = obs_flat[received_indices] + np.random.normal(0, sig_m, size=len(received_indices))



    # Compute the Kalman gain using localized Pb
    D = H_received @ Pb @ H_received.T + R_received
    K = Pb @ H_received.T @ np.linalg.inv(D)

    # State after data assimilation
    stateAfterDA = np.zeros([M, Nlat * Nlon * 2])

    for i in range(M):
        # Innovation for the remaining observations
        innovation = obs_remaining - H_received @ numericalState[i, :]
        stateAfterDA[i, :] = numericalState[i, :] + K @ innovation

    return stateAfterDA


