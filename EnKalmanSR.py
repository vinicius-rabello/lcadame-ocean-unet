# This is base to start the Sr observation, for going to original try Enkalman for base results





import numpy as np
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator


#================================================================================
#================================================================================
#
# def inverse_SMW(ensemble,sig_m,R,Nlat,Nlon,Pb):
#     ensemble_mean = np.mean(ensemble,0)
#     deviations = ensemble - ensemble_mean
#     A = deviations.T  # Transpose to make deviations as columns
#     # print(np.shape(A))
#     AT = deviations
#     N = ensemble.shape[0]
#     PP = (A @ AT) / (N - 1)
#     # Pb = (1/(N-1)) * (deviations.reshape(1,-1)).T @ (deviations.reshape(1,-1))
#     print(PP==Pb)
#     ######  R Inverse  #########################################
#     rInv=np.linalg.inv(R)
#     # Or
#     # rInv = (1/ sig_m ** 2) * np.eye(Nlat * Nlon * 2, Nlat * Nlon * 2)     # Cheaper
#     ################################################################
#     sInv= rInv - rInv @ A @ np.linalg.inv((N-1)*np.eye(N)+ AT @ rInv @ A ) @ AT @ rInv
#     ###############################################
#     return sInv




def inverse_SMW(ensemble,sig_m,R,Nlat,Nlon,Pb,H):    #   Simplified

    ######  R Inverse  #########################################
    rInv=np.linalg.inv(H+Pb@R)
    # Or
    # rInv = (1/ sig_m ** 2) * np.eye(Nlat * Nlon * 2, Nlat * Nlon * 2)     # Cheaper
    ################################################################
    sInv= R@Pb@rInv
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
def EnKFSR(ubi,observations,numericalState,N,M,Nlat,Nlon):


    #####    SR      #########################################





    # The analysis step for the (stochastic) ensemble Kalman filter
    # with virtual observations
    ub = np.mean(ubi,0)   
    Pb = (1/(N-1)) * (ubi - ub.reshape(1,-1)).T @ (ubi - ub.reshape(1,-1))
    # print(np.shape(Pb))
    # print('Inside KF, ub',np.shape(Pb),ub.shape)
    # print(np.shape(ubi),np.shape(observations),np.shape(numericalState))




    # D = np.zeros((2 * Nlat, 2 * Nlat * Nlon))
    # D= np.zeros_like(observations)
    # diagonal=[]
    # diagonals = []
    #
    # # Iterate over the first dimension (2 slices)
    # for i in range(observations.shape[0]):
    # #     # Extract the diagonal from the current slice
    #     diagonal = observations[i][i]
    #     diagonals.append(diagonal)
    # flattened_diagonals = np.array(diagonals).flatten()
    ########################################################################
    # Shape of the 3D matrix
    shape = (46, 68, 2)

    # Initialize the H matrix
    # H = np.ones((2 * Nlat * Nlon, 2 * Nlat * Nlon))
    H = np.eye(2 * Nlat * Nlon)
    # H = np.zeros((2 * Nlat * Nlon, 2 * Nlat * Nlon))

    # Fill the H matrix
    # row_index = 0
    # for i in range(min(Nlat, Nlon)):
    #     # Extract the diagonal indexes from the current slice
    #     indices = [(i, i, k) for k in range(2)]
    #     for j, index in enumerate(indices):
    #         # print('index====', index)
    #         H[row_index, convert_to_1d_index(index, shape)] = 1
    #         row_index += 1



    sig_m= 1e-9
    # sig_m= 0.0000000001 #0.15
    # sig_m= 0.0 #0.15
    # sig_m= 0.0000001 #0.15
    # sig_m= 0.15
    # R = sig_m**2*np.eye(Nlat*Nlon*2,Nlat*Nlon*2)
    # R = sig_m**2*np.eye(2*observations.shape[0],2*observations.shape[0])
    R = sig_m**2*np.eye(2* Nlat * Nlon,2* Nlat * Nlon)







    # print("====================================")
    # cond_numberPb = np.linalg.cond(Pb)
    # print("Condition number of Pb:", cond_numberPb)
    # cond_numberD = np.linalg.cond(D)
    # print("Condition number of Pb:", cond_numberD)
    # # Condition number of Pb: 1.4704044661406397e+21
    # # Condition number of Pb: 2816336159.309645


    ##### Inv ######################################
    # D = H @ Pb @ H.T + R
    # D1=np.linalg.inv(D)
    # D2=np.linalg.pinv(D)
    # D3=inverse_SMW(ubi,sig_m,R,Nlat,Nlon,Pb,H)
    #
    # print(D@D1==H,D@D2==H,D@D3==H)

    # The best inverse is np.linalg.pinv(D)

    # Compute the Kalman gain While R=0 then:
    # D = H @ Pb @ H.T + R= H @ Pb H.T = Pb
    # K = Pb @ H.T @ np.linalg.pinv(D) = Pb @ (Pb)^-1 = I
    # Therefore ====> K=H = I
    # K = H

    # Compute the Kalman gain using H
    D = H @ Pb @ H.T + R
    K = Pb @ H.T @ np.linalg.pinv(D)
    print(np.shape(K),K)



        

    # obsPlusError = np.zeros([Nlat*Nlon*2])
    # obsPlusError = np.zeros([2*observations.shape[0]])
    stateAfterDA = np.zeros([M,Nlat*Nlon*2])
    
    obsPlusError = observations.flatten() #+ np.random.normal(0,sig_m,[2*Nlat*Nlon,])
    # obsPlusError = observations.flatten() + np.random.normal(0,sig_m,[observations.shape[0],])
    # obsPlusError = flattened_diagonals+ np.random.normal(0,sig_m,[2*observations.shape[0],])
    # obsPlusError = flattened_diagonals+ np.random.normal(0,sig_m,[2*observations.shape[0],])




    for i in range(M):
        # stateAfterDA[i,:] = numericalState[i,:] + K @ (obsPlusError[:]-numericalState[i,:])
        innovation = obsPlusError[:] - H @ numericalState[i, :]
        # print(innovation)
        # innovationH = obsPlusError - H @ numericalState[i, :]

        # stateAfterDA[i, :] = numericalState[i, :] + K @ innovation
        stateAfterDA[i, :] = numericalState[i, :] + K @ innovation
    return stateAfterDA
#####################################################################################
