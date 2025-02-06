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


def EnKS(stateAfterDA, K, Pb, H_received, numericalState, M):
    # Initialize smoothed state array
    smoothedState = np.copy(stateAfterDA)

    # Iterate backwards in time for smoothing
    for k in range(M - 2, -1, -1):
        # Compute smoother gain
        Ck = Pb @ H_received.T @ np.linalg.inv(H_received @ Pb @ H_received.T + np.eye(H_received.shape[0]))

        # Update smoothed state
        smoothedState[k, :] = stateAfterDA[k, :] + Ck @ (smoothedState[k + 1, :] - numericalState[k, :])

    return smoothedState

#####################################################################################
# Ensemble Kalman filter
#####################################################################################
def EnKFHV(ubi, observations, numericalState, N, M, Nlat, Nlon,t):
    # The analysis step for the (stochastic) ensemble Kalman filter
    # with virtual observations


    # The analysis step for the (stochastic) ensemble Kalman filter with virtual observations
    obs = np.copy(observations)
    obs = np.full(observations.shape, np.nan)

    # Perform updates for t iterations
    # for t in range(4, 301):  # t ranges from 4 to 300
    #     # Determine the range of rows to update in this iteration
    #     start_row = max(0, t - 4)
    #     end_row = t + 1  # inclusive of t
    if t<164:
        T1=int(t / 5)
        print(T1,T1+14)
    else:
        T1=int(t / 5)-32
        print(T1,T1+14)



    # Update the selected rows
    for j in range(T1-1, T1+14):  # Loop over the selected rows
        for i in range(observations.shape[1]):  # Loop over all columns
            obs[j, i, :] = observations[j, i, :]




    Nlat=obs.shape[0]
    Nlon=obs.shape[1]

    # Flatten the observations and identify the received (non-NaN) indices
    obs_flat = np.array(obs).flatten()
    total_indices = np.arange(len(obs_flat))
    received_indices = np.where(~np.isnan(obs_flat))[0]
    lost_indices = np.where(np.isnan(obs_flat))[0]
    num_lost =  len(obs_flat)-(int((Nlat-5)*Nlon * 2))


    # Determine the number of observations to drop
    # num_lost = int(len(received_indices) * loss_percentage)


    # Randomly select the indices to drop
    # lost_indices = np.random.choice(received_indices, num_lost, replace=False)

    # Set the selected observations to NaN
    # obs_flat[lost_indices] = np.nan

    # Reshape the observations back to the original shape
    obs = obs_flat.reshape(obs.shape)


    ub = np.mean(ubi, 0)
    Pb = (1 / (N - 1)) * (ubi - ub.reshape(1, -1)).T @ (ubi - ub.reshape(1, -1))
    # print('Inside KF, ub',np.shape(Pb),ub.shape)

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

    # stateAfterDA = apply_3D_Var(stateAfterDA, observations, H, R)
    stateAfterDA = apply_3D_Var(stateAfterDA, obs, H_received, R_received)


    return stateAfterDA

    # for i in range(M):
    #     # Innovation for remaining observations
    #     innovation = obs_remaining - H_received @ numericalState[i, :]
    #     stateAfterDA[i, :] = numericalState[i, :] + K @ innovation
    #
    # # Smoothing step using EnKS
    # smoothedState = EnKS(stateAfterDA, K, Pb, H_received, numericalState, M)
    #
    # return smoothedState

# def apply_3D_Var(stateAfterDA, observations, H, R):
#     # Perform 3D-Var update
#     # Assuming linear approximation for simplicity
#     H_T = H.T
#     HT_R_inv = np.linalg.inv(H_T @ np.linalg.inv(R) @ H + np.eye(H.shape[1]))
#     K_3D_Var = np.linalg.inv(R) @ H @ HT_R_inv
#
#     for i in range(stateAfterDA.shape[0]):
#         innovation = observations.flatten() - H @ stateAfterDA[i, :]
#         stateAfterDA[i, :] += K_3D_Var @ innovation
#
#     return stateAfterDA
def apply_3D_Var(stateAfterDA, observations, H_received, R_received):
    # Perform 3D-Var update using partial observations
    H_T = H_received.T
    HT_R_inv = np.linalg.inv(H_T @ np.linalg.inv(R_received) @ H_received + np.eye(H_received.shape[1]))
    K_3D_Var = np.linalg.inv(R_received) @ H_received @ HT_R_inv

    obs_flat = observations.flatten()
    received_indices = np.where(~np.isnan(obs_flat))[0]
    obs_received = obs_flat[received_indices]

    for i in range(stateAfterDA.shape[0]):
        innovation = obs_received - H_received @ stateAfterDA[i, :]
        stateAfterDA[i, :] += K_3D_Var @ innovation

    return stateAfterDA




