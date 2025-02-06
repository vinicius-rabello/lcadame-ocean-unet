import sys
import numpy as np
from DryModel import drymodel

from unet import stn
from EnKalman import EnKF
from qPlotResults import plotResults

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

__version__ = 0.1


#####################################################################################
# U-Net
#####################################################################################
model = stn()
model.compile(loss='mse', optimizer='adam')
model.summary()

model.load_weights('best_weights_lead1.h5')
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
    
    return psi_ensemble_num_new


def oneStep_ML(psi_ensemble_ml0,Nlat,Nlon):
    N = psi_ensemble_ml0.shape[0]
    ### Evolve ensembles with Unet##################################
    psi_ensemble_ml_new = np.zeros_like(psi_ensemble_ml0)
    for k in range(0,N):     
      E = psi_ensemble_ml0[k,:].reshape([Nlat, Nlon, 2])
      E2 = auxLat2Lon(E.reshape([Nlat,Nlon,2]))
      pred_temp = (model.predict(E2.flatten().reshape([1,Nlon,Nlat,2])))
      P2 = auxLat2Lon(pred_temp.flatten().reshape([Nlon,Nlat,2]))
      psi_ensemble_ml_new[k,:] = P2.flatten() #pred_temp.copy()
       
    aux_psi_ensemble_new_denorm=np.zeros([N,2*Nlat*Nlon])
    for k in range(0,N):
        uu = psi_ensemble_ml_new[k,:].reshape([Nlat, Nlon, 2])
        uu[:,:,0] = uu[:,:,0]*STD_L1+MEAN_L1
        uu[:,:,1] = uu[:,:,1]*STD_L2+MEAN_L2
        aux_psi_ensemble_new_denorm[k,:] = uu.flatten()
   
    return psi_ensemble_ml_new,aux_psi_ensemble_new_denorm

#####################################################################################
def calcMetrics(psi,psi_exact,psi_exact_average):
  Ek1 = np.linalg.norm(psi[:,:,0]-psi_exact[:,:,0])/np.max(psi_exact[:,:,0])
  Ek2 = np.linalg.norm(psi[:,:,1]-psi_exact[:,:,1])/np.max(psi_exact[:,:,1])

  den11 = np.sum(np.multiply(psi[:,:,0]-psi_exact_average[:,:,0],psi[:,:,0]-psi_exact_average[:,:,0]))
  den12 = np.sum(np.multiply(psi_exact[:,:,0]-psi_exact_average[:,:,0],psi_exact[:,:,0]-psi_exact_average[:,:,0]))
  den21 = np.sum(np.multiply(psi[:,:,1]-psi_exact_average[:,:,1],psi[:,:,1]-psi_exact_average[:,:,1]))
  den22 = np.sum(np.multiply(psi_exact[:,:,1]-psi_exact_average[:,:,1],psi_exact[:,:,1]-psi_exact_average[:,:,1]))

  Acc1 = np.sum(np.multiply(psi[:,:,0]-psi_exact_average[:,:,0],psi_exact[:,:,0]-psi_exact_average[:,:,0]))/(den11*den12)**0.5
  Acc2 = np.sum(np.multiply(psi[:,:,1]-psi_exact_average[:,:,1],psi_exact[:,:,1]-psi_exact_average[:,:,1]))/(den21*den22)**0.5

  return np.array([Ek1,Ek2,Acc1,Acc2])

#####################################################################################

### Load dataset for truth and Obs #########

Lx = 96  #46. #size of x -- stick to multiples of 10
Ly = 192 #68.

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
R = sig_m**2*np.eye(Nlat*Nlon*2,Nlat*Nlon*2)

#############################################################################################################
# Prepare Observations from exact simulation
#############################################################################################################
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


#############################################################################################################
N = 10 #2000 #int(sys.argv[1])
M=10+1 # last one for the exact solution

print('number of numerical ens',M)
print('number of DD ens',N)

sig_b= 0.1
B = sig_b**2*np.eye(2*Nlat*Nlon,Nlat*Nlon*2)
Q = 0.0*np.eye(2*Nlat*Nlon,Nlat*Nlon*2)
#############################################################################################################


#############################################################################################################
# Memory Allocation 
#############################################################################################################
T = 300 # this is days

E_tr = np.zeros([Nlat,Nlon,2])
pred_temp_tr = np.zeros([Nlon,Nlat,2])

psi_ensemble=np.zeros([N,2*Nlat*Nlon])
psi_ensemble_new=np.zeros([N,2*Nlat*Nlon])
psi_ensemble_new_denorm=np.zeros([N,2*Nlat*Nlon])

psi_ensemble_numerical=np.zeros([M,2*Nlat*Nlon])
psi_ensemble_numerical_new=np.zeros([M,2*Nlat*Nlon])

psi_updated=np.zeros([T+1,Nlat,Nlon,2])
#Pb_updated=np.zeros([Nlat*Nlon*2,Nlat*Nlon*2])

psi_exact_average=np.zeros([Nlat,Nlon,2])

metrics = np.zeros((T+1,4))
#############################################################################################################

#############################################################################################################
# Initial condition
#############################################################################################################
# this is for machine learning
psi0_ML = np.zeros([Nlat, Nlon, 2])
psi0_ML[:,:,0]=(psi0[:,:,0]-MEAN_L1)/STD_L1
psi0_ML[:,:,1]=(psi0[:,:,1]-MEAN_L2)/STD_L2

for k in range(0,N):
    psi_ensemble[k,:] = (psi0.flatten()+np.random.normal(0,sig_b,[2*Nlat*Nlon,]))
       
for k in range (0,M-1):
    psi_ensemble_numerical[k,:] = (psi0.flatten()+np.random.normal(0,sig_b,[2*Nlat*Nlon,])) 

psi_ensemble_numerical[M-1,:] = psi0.flatten()
#############################################################################################################


#############################################################################################################
# Time Advance
#############################################################################################################
count,t = 0,0
while (t<T+1):   
    #========================================================================================================
    if (t>0 and (t+1) % DA_cycles ==0):
        #------------------------------------------------------------------------------------------------------
        # We receive observation here
        #------------------------------------------------------------------------------------------------------
        #psi_ensemble,psi_ensemble_numerical,psi_ensemble_new_denorm = oneStep(psi_ensemble,psi_ensemble_numerical,Nlat,Nlon)
        psi_ensemble_numerical = oneStep_Numerical(psi_ensemble_numerical,Nlat,Nlon)
        psi_ensemble,psi_ensemble_new_denorm = oneStep_ML(psi_ensemble,Nlat,Nlon)

        #### Start ENKF, Pass determinstic state as well frm NM ################    
        print('Starting KF')
        #P = (1/(N-1)) * (psi_ensemble_new_denorm - (np.mean(psi_ensemble_new_denorm,0)).reshape(1,-1)) @ (psi_ensemble_new_denorm - (np.mean(psi_ensemble_new_denorm,0)).reshape(1,-1)).T
        psi_ensemble_numerical, _, _ = EnKF(psi_ensemble_new_denorm,obs[count,:,:,:],psi_ensemble_numerical,N,M,Nlat,Nlon)

        #Pb_updated=Pb_updated + Pb
        #psi_ensemble_numerical = np.asarray(psi_ensemble_numerical)
        count=count+1

        ####### Update with mean of ensembles from ENKF output ##############################
        psi_updated[t,:,:,:]=(np.mean(psi_ensemble_numerical,1)).reshape([Nlat,Nlon,2])
    
        ### Restart DD ensembles based on new numerical update #########################################
        for k in range(0,N):
            psi_ensemble_new[:,k] = ((np.mean(psi_ensemble_numerical,1)).reshape([Nlat,Nlon,2]).flatten()+np.random.normal(0,sig_b,[2*Nlat*Nlon,]))
 
        #------------------------------------------------------------------------------------------------------
    else:
        #------------------------------------------------------------------------------------------------------
        # Cycles without observations
        #------------------------------------------------------------------------------------------------------
        #psi_ensemble,psi_ensemble_numerical,psi_ensemble_new_denorm = oneStep(psi_ensemble,psi_ensemble_numerical,Nlat,Nlon)
        psi_ensemble_numerical = oneStep_Numerical(psi_ensemble_numerical,Nlat,Nlon)

        ############### Evolve with numerical solver ##########################################################
        psi_updated[t,:,:,:] = (np.mean(psi_ensemble_numerical[0:-1],0)).reshape([Nlat,Nlon,2])
        psi_exact_average = psi_exact_average + psi_ensemble_numerical[-1].reshape([Nlat,Nlon,2])
        res = calcMetrics(psi_updated[t,:,:,:],psi_ensemble_numerical[-1].reshape([Nlat,Nlon,2]),psi_exact_average/(t+1))
        metrics[t,:] = res
        #------------------------------------------------------------------------------------------------------
        ############### Plot results ##########################################################################
        plotResults(Lx,Ly,psi_updated[t,:,:,:],psi_ensemble_numerical[-1].reshape([Nlat,Nlon,2]),metrics,t,T)
        #------------------------------------------------------------------------------------------------------
    #========================================================================================================
    t=t+1
    print('Out of '+str(T+1),t)
#############################################################################################################
# Time Advance
#############################################################################################################


#Pb_updated=Pb_updated/(count-1)
#savenc(psi_updated, x, y, 'Psi_updated_hybridDL_combined_DDrestartM' + str(M) + 'T'+str(T)+'ens'+str(N)+'.nc')
#np.savetxt('analysis_cov_DDrestart_M' +str(M)+'DL' +str(N)+'T'+str(T)+'.csv',Pb_updated,delimiter=',')

