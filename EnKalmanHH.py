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
def EnKFHH(ubi, observations, numericalState, N, M, Nlat, Nlon):
    # The analysis step for the (stochastic) ensemble Kalman filter
    # with virtual observations
    ub = np.mean(ubi, 0)
    Pb = (1 / (N - 1)) * (ubi - ub.reshape(1, -1)).T @ (ubi - ub.reshape(1, -1))
    # print('Inside KF, ub',np.shape(Pb),ub.shape)

    # Shape of the 3D matrix
    shape = (46, 68, 2)

    # D = np.zeros((2 * Nlat, 2 * Nlat * Nlon))
    # D= np.zeros_like(observations)
    diagonal=[]
    diagonals = []

    # Iterate over the first dimension (2 slices)
    for i in range(observations.shape[0]):
    #     # Extract the diagonal from the current slice
        diagonal = observations[i][0]
        diagonals.append(diagonal)
    flattened_diagonals = np.array(diagonals).flatten()
    # ########################################################################
    # # Shape of the 3D matrix
    # shape = (46, 68, 2)
    #
    def convert_to_1d_index(index, shape):
        ind = index[0] * (shape[1] * shape[2]) + index[1] * shape[2] + index[2]
        # print(ind)
        return ind

    # Initialize the H matrix
    # Nlat = 46  # Example value, replace with the actual value
    # Nlon = 68  # Example value, replace with the actual value

    # size = 2 * Nlat * Nlon
    # H = np.eye(size)
    H = np.zeros((2 * Nlat * Nlon, 2 * Nlat * Nlon))
    # H = np.zeros((2 * Nlat, 2 * Nlat * Nlon))

    # Shape of the 3D matrix
    # shape = (46, 68, 2)

    # Fill the H matrix
    row_index = 0
    for i in range(min(Nlat, Nlon)):
        # Extract the diagonal indexes from the current slice
        indices = [(i, i, k) for k in range(2)]
        for j, index in enumerate(indices):
            # print('index====', index)
            H[row_index, 0] = 1
            row_index += 1

    # Initialize the H matrix
    H = np.eye((2 * Nlat * Nlon, 2 * Nlat * Nlon))



    sig_m = 0.15
    R = sig_m**2*np.eye(2*observations.shape[0],2*observations.shape[0])
    # R = sig_m**2*np.eye((2 * Nlat * Nlon, 2 * Nlat * Nlon))
    # compute Jacobian of observation operator at ub
    # Dh = np.eye(2*Nlat*Nlon,2*Nlat*Nlon)
    # compute Kalman gain
    # D = Dh@B@Dh.T + R
    # K = B @ Dh.T @ np.linalg.inv(D)
    # D = Pb + R


    # K = Pb @ approximate_inverse_cg(D) #np.linalg.inv(D)
    # K = Pb @ np.linalg.inv(D)

    # Compute the Kalman gain using H
    D = H @ Pb @ H.T + R
    K = Pb @ H.T @ np.linalg.inv(D)

    # print('==>Inside KF, K',np.shape(K))

    # update state of the system
    obsPlusError = np.zeros([Nlat * Nlon * 2])
    stateAfterDA = np.zeros([M, Nlat * Nlon * 2])


    # obsPlusError = observations.flatten() + np.random.normal(0, sig_m, [2 * Nlat * Nlon, ])
    obsPlusError = flattened_diagonals+ np.random.normal(0,sig_m,[2*observations.shape[0],])


    for i in range(M):
        stateAfterDA[i, :] = numericalState[i, :] + K @ (obsPlusError[:] - H @  numericalState[i, :])

    # remove observation on psi2
    # stateAfterDA = stateAfterDA.reshape((M, Nlat, Nlon, 2))
    # for i in range(M):
    #     uu = numericalState[i, :].reshape((Nlat, Nlon, 2))
    #     stateAfterDA[i, :, :, 1] = uu[:, :, 1]
    # stateAfterDA = stateAfterDA.reshape((M, Nlat * Nlon * 2))

    return stateAfterDA

