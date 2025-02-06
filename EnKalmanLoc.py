import numpy as np
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

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



#####################################################################################
# Ensemble Kalman filter
#####################################################################################
def EnKFLoc(ubi,observations,numericalState,N,M,Nlat,Nlon):
     
    # The analysis step for the (stochastic) ensemble Kalman filter 
    # with virtual observations
    ub = np.mean(ubi,0)   
    Pb = (1/(N-1)) * (ubi - ub.reshape(1,-1)).T @ (ubi - ub.reshape(1,-1))
    print(np.shape(Pb))


    ###   Localization               ###############################################################################3

    # Define the localization matrix based on the size of Pb
    n = Pb.shape[0]  # size of Pb, which is 2496
    localization_matrix = np.zeros((n, n))

    # Set up Gaspari-Cohn function coefficients (assuming simple for illustration)
    # These values would depend on your actual localization radius and the Gaspari-Cohn function.
    # Here, we're assuming it decreases with distance and affects only the 5 diagonals.

    # For simplicity, let's assume we define a constant weight for each diagonal:
    diag_main = 1.0  # Main diagonal
    diag_1 = 0.8  # First off-diagonals
    diag_2 = 0.4  # Second off-diagonals

    # Fill in the diagonals
    np.fill_diagonal(localization_matrix, diag_main)  # Main diagonal
    np.fill_diagonal(localization_matrix[1:], diag_1)  # First upper diagonal
    np.fill_diagonal(localization_matrix[:, 1:], diag_1)  # First lower diagonal
    np.fill_diagonal(localization_matrix[2:], diag_2)  # Second upper diagonal
    np.fill_diagonal(localization_matrix[:, 2:], diag_2)  # Second lower diagonal

    # Multiply the covariance matrix with the localization matrix element-wise
    Pb_localized = Pb * localization_matrix  # Element-wise multiplication

    #########################################################################################
    
    sig_m= 0.15 
    R = sig_m**2*np.eye(Nlat*Nlon*2,Nlat*Nlon*2)
    # compute Jacobian of observation operator at ub
    #Dh = np.eye(2*Nlat*Nlon,2*Nlat*Nlon)
    # compute Kalman gain
    #D = Dh@B@Dh.T + R    
    #K = B @ Dh.T @ np.linalg.inv(D)
    D = Pb + R
    # D = Pb_localized + R
    print(np.shape(D),"D",np.shape(Pb))

    #K = Pb @ approximate_inverse_cg(D) #np.linalg.inv(D)
    K = Pb @ np.linalg.inv(D)
    # K = Pb_localized @ np.linalg.inv(D)

    print('==>Inside KF, K',np.shape(K))    

    # update state of the system
    obsPlusError = np.zeros([M,Nlat*Nlon*2])
    stateAfterDA = np.zeros([M,Nlat*Nlon*2])

    obsPlusError = observations.flatten() + np.random.normal(0, sig_m, [2 * Nlat * Nlon, ])
    
    for i in range(M):
        # create virtual observations
        # obsPlusError[i,:] = observations.flatten() + np.random.normal(0,sig_m,[2*Nlat*Nlon,])
        # compute analysis ensemble
        stateAfterDA[i,:] = numericalState[i,:] + K @ (obsPlusError-numericalState[i,:])
        # if i>=M:
        #     stateAfterDA[i, :] = numericalState[i, :] #+ K @ (obsPlusError - numericalState[i, :])

        
    # print('Inside KF, uai',np.shape(stateAfterDA))
    # compute the mean of analysis ensemble
    ua = np.mean(stateAfterDA,0)

    print('Inside KF, ua',np.shape(ua))
    # compute analysis error covariance matrix
    nP = (1/(M-1)) * (stateAfterDA - ua.reshape(1,-1)) @ (stateAfterDA - ua.reshape(1,-1)).T
    return stateAfterDA
#####################################################################################
