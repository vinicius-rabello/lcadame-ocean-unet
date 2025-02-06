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

def EnKF3(ubi, observations, numericalState, N, M, Nlat, Nlon):
    # The analysis step for the (stochastic) ensemble Kalman filter with virtual observations
    obs = np.copy(observations)
    # obs=np.zeros_like(observations)
    # diagonal=[]
    # diagonals = []
    #
    # # Iterate over the first dimension (2 slices)
    # for i in range(observations.shape[1]):
    # #     # Extract the diagonal from the current slice
    #     diagonal = observations[0][i]
    #     diagonals.append(diagonal)
    # flattened_diagonals = np.array(diagonals).flatten()
    #
    # obs=diagonals



    ub = np.mean(ubi, 0)
    Pb = (1/(N-1)) * (ubi - ub.reshape(1, -1)).T @ (ubi - ub.reshape(1, -1))

    # Initialize the H matrix
    shape = (46, 68, 2)
    size = 2 * Nlat * Nlon
    H = np.eye(size)

    sig_m = 0.15
    R = sig_m**2 * np.eye(2 * Nlat * Nlon, 2 * Nlat * Nlon)


    # Flatten the observations and identify the received (non-NaN) indices
    obs[:, :, 1] = np.nan
    obs_flat = np.array(obs).flatten()

    received_indices = np.where(~np.isnan(obs_flat))[0]
    lost_indices = np.where(np.isnan(obs_flat))[0]

    # Create a reduced observation matrix H_received for the received observations
    H_received = H[received_indices, :]

    # Reduce the R matrix to correspond to the received observations
    R_received = R[np.ix_(received_indices, received_indices)]

    # Create the received observations vector
    obs_received = obs_flat[received_indices] + np.random.normal(0, sig_m, size=len(received_indices))

    # Compute the Kalman gain using H_received
    D = H_received @ Pb @ H_received.T + R_received
    K = Pb @ H_received.T @ np.linalg.inv(D)

    # State after data assimilation
    stateAfterDA = np.zeros([M, Nlat * Nlon * 2])

    for i in range(M):
        # Innovation for the received observations
        innovation = obs_received - H_received @ numericalState[i, :]
        stateAfterDA[i, :] = numericalState[i, :] + K @ innovation
    return stateAfterDA

# # Example parameters (these should be adjusted for your specific problem)
# ubi = np.random.rand(10, 2 * 46 * 68)  # Example ensemble
# observations = np.random.rand(2, 46, 68)  # Example observations
# numericalState = np.random.rand(10, 2 * 46 * 68)  # Example numerical state
# H = np.ones((2 * 46 * 68, 2 * 46 * 68))  # Example H matrix
# N = 10  # Ensemble size
# M = 10  # Number of members
# Nlat = 46  # Number of latitudes
# Nlon = 68  # Number of longitudes
#
# # Introduce NaN values for testing
# observations[0, 1, 0] = np.nan
# observations[1, 2, 1] = np.nan

# stateAfterDA = EnKF(ubi, observations, numericalState, H, N, M, Nlat, Nlon)
# print(stateAfterDA)
