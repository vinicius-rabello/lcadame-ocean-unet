import sys
import numpy as np
from DryModel import drymodel

from EnKalman import EnKF
from EnKalmanHD import EnKFHD

# from EnKalmanLoc import EnKFLoc
from EnKalmanHP import EnKFHP_Loc
from EnKalman2 import EnKF2
from EnKalmanHupdate import EnKF3
from EnKalman4 import EnKF4
from qPlotResults import plotResults
import os
import matplotlib.pyplot as plt

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# Ensure the directory exists
output_dir = 'C:\\Users\\rezaa\\Desktop\\Ocean\\H-Code-20240617\\reza(5)\\imagess'
# Ensure the directory exists before saving the image
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#####################################################################################
MEAN_L1=0
STD_L1=1
MEAN_L2=0
STD_L2=1

#####################################################################################
def auxLat2Lon(inp):
   Nlon = inp.shape[1]
   Nlat = inp.shape[0]
   pred_temp_layer1 = inp[:,:,0]
   pred_temp_layer2 = inp[:,:,1]

   pred_temp_tr = np.zeros([Nlon,Nlat,2])
   pred_temp_tr[:,:,0] = np.transpose(pred_temp_layer1)
   pred_temp_tr[:,:,1] = np.transpose(pred_temp_layer2)
   return pred_temp_tr

#####################################################################################
# oneStep time progress using both machine learning and numerical integration
#####################################################################################
def oneStep_Numerical(psi_ensemble_num0,Nlat,Nlon):
    M = psi_ensemble_num0.shape[0]
    psi_ensemble_num_new = np.zeros_like(psi_ensemble_num0)
    #--- Evolve numerical ensembles  with numerical solver
    for k in range(0,M):
      psi_ensemble_num_new[k,:] = (drymodel(psi_ensemble_num0[k,:].reshape([Nlat,Nlon,2]),1.0)).flatten()

    ##########################
    aux_psi_ensemble_new_denorm=np.zeros([M,2*Nlat*Nlon])
    for k in range(0,M):
       uu = psi_ensemble_num_new[k,:].reshape([Nlat, Nlon, 2])
       uu[:,:,0] = uu[:,:,0]*STD_L1+MEAN_L1
       uu[:,:,1] = uu[:,:,1]*STD_L2+MEAN_L2
       aux_psi_ensemble_new_denorm[k,:] = uu.flatten()
    
    return psi_ensemble_num_new #,aux_psi_ensemble_new_denorm
#======================================================================================

# def oneStep_ML(psi_ensemble_ml0,Nlat,Nlon):
#    N = psi_ensemble_ml0.shape[0]
#    ### Evolve ensembles with Unet##################################
#    psi_ensemble_ml_new = np.zeros_like(psi_ensemble_ml0)
#    for k in range(0,N):
#      E = psi_ensemble_ml0[k,:].reshape([Nlat, Nlon, 2])
#      E2 = auxLat2Lon(E.reshape([Nlat,Nlon,2]))
#      pred_temp = (model.predict(E2.flatten().reshape([1,Nlon,Nlat,2])))
#      P2 = auxLat2Lon(pred_temp.flatten().reshape([Nlon,Nlat,2]))
#      psi_ensemble_ml_new[k,:] = P2.flatten() #pred_temp.copy()
#
#    aux_psi_ensemble_new_denorm=np.zeros([N,2*Nlat*Nlon])
#    for k in range(0,N):
#        uu = psi_ensemble_ml_new[k,:].reshape([Nlat, Nlon, 2])
#        uu[:,:,0] = uu[:,:,0]*STD_L1+MEAN_L1
#        uu[:,:,1] = uu[:,:,1]*STD_L2+MEAN_L2
#        aux_psi_ensemble_new_denorm[k,:] = uu.flatten()
#
#    return psi_ensemble_ml_new,aux_psi_ensemble_new_denorm

#####################################################################################
def calcMetrics(psi,psi_exact,psi_exact_average):
  Ek1 = np.linalg.norm(psi[:,:,0]-psi_exact[:,:,0]) /np.max(np.absolute(psi_exact[:,:,0]))
  Ek2 = np.linalg.norm(psi[:,:,1]-psi_exact[:,:,1]) /np.max(np.absolute(psi_exact[:,:,1]))

  den11 = np.sum(np.multiply(psi[:,:,0]-psi_exact_average[:,:,0],psi[:,:,0]-psi_exact_average[:,:,0]))
  den12 = np.sum(np.multiply(psi_exact[:,:,0]-psi_exact_average[:,:,0],psi_exact[:,:,0]-psi_exact_average[:,:,0]))
  den21 = np.sum(np.multiply(psi[:,:,1]-psi_exact_average[:,:,1],psi[:,:,1]-psi_exact_average[:,:,1]))
  den22 = np.sum(np.multiply(psi_exact[:,:,1]-psi_exact_average[:,:,1],psi_exact[:,:,1]-psi_exact_average[:,:,1]))

  den11 = max(1.0e-12,den11)
  den12 = max(1.0e-12,den12)
  den21 = max(1.0e-12,den21)
  den22 = max(1.0e-12,den22)

  Acc1 = np.sum(np.multiply(psi[:,:,0]-psi_exact_average[:,:,0],psi_exact[:,:,0]-psi_exact_average[:,:,0]))/(den11*den12)**0.5
  Acc2 = np.sum(np.multiply(psi[:,:,1]-psi_exact_average[:,:,1],psi_exact[:,:,1]-psi_exact_average[:,:,1]))/(den21*den22)**0.5

  # return np.array([Ek1,Ek2,Acc1,Acc2])
  # Calculate MSE for psi[:,:,0] and psi[:,:,1]
  MSE1 = np.mean((psi[:, :, 0] - psi_exact[:, :, 0]) ** 2)
  MSE2 = np.mean((psi[:, :, 1] - psi_exact[:, :, 1]) ** 2)

  return np.array([Ek1, Ek2, Acc1, Acc2, MSE1, MSE2])
#####################################################################################

### Load dataset for truth and Obs #########

Lx = 46 #96  #46. #size of x -- stick to multiples of 10
Ly = 68 #192 #68.

psi = np.load(f'ICs/oneYear.npy')

#MEAN_L1 = np.mean(psi[:,0,:,:].flatten())
#STD_L1  = np.std(psi[:,0,:,:].flatten())

#MEAN_L2 = np.mean(psi[:,1,:,:].flatten())
#STD_L2  = np.std(psi[:,1,:,:].flatten())

Nlat=np.size(psi,1) #np.size(psi,2)
Nlon=np.size(psi,2) #np.size(psi,3)

print('size of Nlat',Nlat)
print('size of Nlon',Nlon)
print('shape of psi',psi.shape)

######## Emulate Observation with noise ########

sig_m= 0.15  # standard deviation for measurement noise
#R = sig_m**2*np.eye(Nlat*Nlon*2,Nlat*Nlon*2)

#############################################################################################################
# Prepare Observations from exact simulation
#####################################     Observations      ########################################################################
DA_cycles=int(5)
obs=np.zeros([int(np.size(psi,0)/DA_cycles),Nlat,Nlon,2])
obs_count=0
for k in range(DA_cycles,np.size(psi,0),DA_cycles):
    obs[obs_count,:,:,:]=psi[k,:,:,:]     
    obs_count=obs_count+1
#############################################################################################################


########### Start initial condition ##########
psi0 = psi[0,:,:,:]
##############################################


H = np.eye(2*Nlat*Nlon,2*Nlat*Nlon)

#############################################################################################################
N = 10 #2000 #int(sys.argv[1])
M=50 # last one for the exact solution

print('number of numerical ens',M)
print('number of DD ens',N)

sig_b= 0.8
#B = sig_b**2*np.eye(2*Nlat*Nlon,Nlat*Nlon*2)
#Q = 0.0*np.eye(2*Nlat*Nlon,Nlat*Nlon*2)
#############################################################################################################


#############################################################################################################
# Memory Allocation 
#############################################################################################################
T = 300 # this is days

#psi_ensemble=np.zeros([N,2*Nlat*Nlon])
#psi_ensemble_new=np.zeros([N,2*Nlat*Nlon])
#psi_ensemble_new_denorm=np.zeros([N,2*Nlat*Nlon])


psi_num  = np.zeros([M,2*Nlat*Nlon])
psi_ML   = np.zeros([N,2*Nlat*Nlon])
psi_true = np.zeros([1,2*Nlat*Nlon])

psi_updated=np.zeros([T+1,Nlat,Nlon,2])

psi_true_average=np.zeros([Nlat,Nlon,2])

metrics = np.zeros((T+1,6))
#############################################################################################################

#############################################################################################################
# Initial condition
#############################################################################################################      
for k in range (0,M):
    psi_num[k,:] = (psi0.flatten()+np.random.normal(0,sig_b,[2*Nlat*Nlon,])) 

#the last one is for the exact solution
psi_true[0,:] = psi0.flatten()
#############################################################################################################

#############################################################################################################
# Time Advance
#############################################################################################################
total_MSE1 = 0
total_MSE2 = 0

count,t = 0,0
while (t<T+1):  
    #========================================================================================================
    specialStep = 0
    #========================================================================================================
    # one step of the true solution
    psi_true = oneStep_Numerical(psi_true,Nlat,Nlon)
    psi_true_average = psi_true_average + psi_true[0].reshape([Nlat,Nlon,2])
    # one step of numerical integration of our ensemble 
    psi_num = oneStep_Numerical(psi_num,Nlat,Nlon)
    #========================================================================================================
    if (t>0 and (t+1) % DA_cycles ==0):
        #------------------------------------------------------------------------------------------------------
        # We receive observation here
        #------------------------------------------------------------------------------------------------------
        #Pay attention to N & M here, because we use numerical ensemble I put M & M as input
        psi_num = EnKFHD(psi_num, psi_true[0].reshape([Nlat, Nlon, 2]), psi_num, N, M, Nlat, Nlon)

        # if t==19:
        #     psi_num = EnKF(psi_num,psi_true[0].reshape([Nlat,Nlon,2]),psi_num,H,M,M,Nlat,Nlon)
        # else:
        #     psi_num = EnKF(psi_num,psi_true[0].reshape([Nlat,Nlon,2]),psi_num,H,M,M,Nlat,Nlon)
        specialStep = 1
    #========================================================================================================
    # Averaging and metrics
    psi_updated[t,:,:,:] = (np.mean(psi_num,0)).reshape([Nlat,Nlon,2])
    res = calcMetrics(psi_updated[t,:,:,:],psi_true[0].reshape([Nlat,Nlon,2]),psi_true_average/(t+1))    
    metrics[t,:] = res
    plotResults(Lx,Ly,psi_updated[t,:,:,:],psi_true[0].reshape([Nlat,Nlon,2]),metrics,t,specialStep)
    #========================================================================================================
    # print(metrics[t, 4] / (t+1))
    MSE1=metrics[t, 4]
    MSE2=metrics[t, 5]
    total_MSE1 += MSE1
    total_MSE2 += MSE2
    t=t+1

    if specialStep==0:
        print('Day ',t,' of '+str(T+1),'    ',res)

    else:
        print('Day ',t,' of '+str(T+1),' DA ',res)

# Calculate the mean of MSE1 values
mean_MSE1 = total_MSE1 / (T+1)
mean_MSE2 = total_MSE2 / (T+1)

# Print the mean MSE1
print(f"Mean MSE1 over {T+1} timesteps: {mean_MSE1}")
print(f"Mean MSE1 over {T+1} timesteps: {mean_MSE2}")
#############################################################################################################
# MSE
# print()
#############################################################################################################

