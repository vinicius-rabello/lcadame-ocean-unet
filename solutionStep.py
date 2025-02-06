import numpy as np
from Drymodel import drymodel

MEAN_L1=0
STD_L1=1
MEAN_L2=0
STD_L2=1

#####################################################################################
# oneStep time progress using both machine learning and numerical integration
#####################################################################################
def oneStep(psi_ensemble_ml0,psi_ensemble_num0,Nlat,Nlon):
    N = psi_ensemble_ml0.shape[0]
    M = psi_ensemble_num0.shape[0]
    ### Evolve ensembles with Unet##################################
    psi_ensemble_ml_new = np.zeros_like(psi_ensemble_ml0)
    psi_ensemble_num_new = np.zeros_like(psi_ensemble_num0)
    for k in range(0,N):     
      E = psi_ensemble_ml0[k,:].reshape([Nlat, Nlon, 2])

      pred_temp = (model.predict(E_tr.reshape([1, Nlat,Nlon,2]))).reshape([Nlat, Nlon, 2])
      pred_temp_layer1 = pred_temp[:,:,0]
      pred_temp_layer2 = pred_temp[:,:,1]

      pred_temp_tr = np.zeros([Nlon,Nlat,2])
      pred_temp_tr[:,:,0] = np.transpose(pred_temp_layer1)
      pred_temp_tr[:,:,1] = np.transpose(pred_temp_layer2)

      psi_ensemble_ml_new[:,k] = pred_temp_tr.flatten()
     
    #### Evolve numerical ensembles  with numerical solver ####################
    for k in range(0,M):
      psi_ensemble_num_new[k,:] = (drymodel(psi_ensemble_num0[k,:].reshape([Nlat,Nlon,2]),1.0)).flatten()

    
    aux_psi_ensemble_new_denorm=np.zeros([N,2*Nlat*Nlon])
    for k in range(0,N):
        uu = psi_ensemble_ml_new[k,:].reshape([Nlat, Nlon, 2])
        uu[:,:,0] = uu[:,:,0]*STD_L1+MEAN_L1
        uu[:,:,1] = uu[:,:,1]*STD_L2+MEAN_L2
        aux_psi_ensemble_new_denorm[k,:] = uu.flatten()
   
    return psi_ensemble_ml_new,psi_ensemble_num_new,aux_psi_ensemble_new_denorm

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
