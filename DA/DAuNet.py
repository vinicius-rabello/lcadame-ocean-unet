from models.drymodel import drymodel
from models.stn import model
import numpy as np

model.load_weights("models/weights/G_46_68.weights.h5")

####################################################################################
# oneStep time progress using both machine learning and numerical integration
#####################################################################################
def oneStep_Numerical(psi_ensemble_num0, Nlat, Nlon):
    M = psi_ensemble_num0.shape[0]
    psi_ensemble_num_new = np.zeros_like(psi_ensemble_num0)

    # --- Evolve numerical ensembles  with numerical solver
    for k in range(0, M):
        psi_ensemble_num_new[k, :] = (
            drymodel(psi_ensemble_num0[k, :].reshape([Nlat, Nlon, 2]), 1.0)
        ).flatten()
    return psi_ensemble_num_new

#####################################################################################
def oneStep_ML(psi_ensemble_num0, Nlat, Nlon, numML, mean, var, sig):
    M = psi_ensemble_num0.shape[0]
    print(M, "===============", numML)
    inputEnsemble = np.zeros((M * numML, Nlat * Nlon * 2))
    resultEnsemble = np.zeros((M * numML, Nlat * Nlon * 2))
    # generating new members for ML part
    for k in range(0, M):
        for n in range(0, numML):
            inputEnsemble[k * numML + n, :] = psi_ensemble_num0[
                k, :
            ] + np.random.normal(
                0,
                sig,
                [
                    2 * Nlat * Nlon,
                ],
            )
    # standardize ML ensemble
    for k in range(0, M):
        for n in range(0, numML):
            uu = inputEnsemble[k * numML + n, :].reshape([Nlat, Nlon, 2])
            uu[:, :, 0] = (uu[:, :, 0] - mean[0]) / var[0] ** 0.5
            uu[:, :, 1] = (uu[:, :, 1] - mean[1]) / var[1] ** 0.5
            inputEnsemble[k * numML + n, :] = uu.flatten()
    # call ML model
    results = model.predict(inputEnsemble.reshape(M * numML, Nlat, Nlon, 2))

    # recover results
    for k in range(0, M):
        for n in range(0, numML):
            uu = results[k * numML + n, :].reshape([Nlat, Nlon, 2])
            uu[:, :, 0] = uu[:, :, 0] * var[0] ** 0.5 + mean[0]
            uu[:, :, 1] = uu[:, :, 1] * var[1] ** 0.5 + mean[1]
            resultEnsemble[k * numML + n, :] = uu.flatten()

    return resultEnsemble