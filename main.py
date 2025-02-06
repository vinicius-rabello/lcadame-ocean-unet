import numpy as np

# from keras.models import Model
from KalmanFilters.EnKalman import EnKF
from DA.DAuNet import oneStep_Numerical, oneStep_ML
from utils.plotResults import plotResults
from utils.calcMetrics import calcMetrics
import os

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Define the directory and file path
dir_path = r"./images"
# file_path = os.path.join(dir_path, "Solver_000000.png")

# Ensure the directory exists
if not os.path.exists(dir_path):
    try:
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created.")
    except Exception as e:
        print(f"Failed to create directory {dir_path}: {e}")
else:
    print(f"Directory {dir_path} already exists.")

# Verify the directory exists by listing its contents
print(f"Contents of {dir_path}: {os.listdir(dir_path)}")


mean_ = np.array([0.00025023, -0.00024681])
var_ = np.array([9.9323115, 0.18261143])

#######################################################################






### Load dataset for truth and Obs #########


Lx = 46  # 96  #46. #size of x -- stick to multiples of 10
Ly = 68  # 192 #68.
psi = np.load("data/oneYear.npy")
print(np.shape(psi), "*****")
Nlat = np.size(psi, 1)  # np.size(psi,2)
Nlon = np.size(psi, 2)  # np.size(psi,3)

print("size of Nlat", Nlat)
print("size of Nlon", Nlon)
print("shape of psi", psi.shape)

######## Emulate Observation with noise ########

sig_m = 0.15  # standard deviation for measurement noise

#############################################################################################################
# Prepare Observations from exact simulation
#############################################################################################################
DA_cycles = int(5)
obs = np.zeros([int(np.size(psi, 0) / DA_cycles), Nlat, Nlon, 2])
obs_count = 0
for k in range(DA_cycles, np.size(psi, 0), DA_cycles):
    obs[obs_count, :, :, :] = psi[k, :, :, :]
    obs_count = obs_count + 1
#############################################################################################################


########### Start initial condition ##########
psi0 = psi[0, :, :, :]
# print(psi0.shape,'================')
##############################################


#############################################################################################################
N = 20  # 2000 #int(sys.argv[1])
M = 20  # last one for the exact solution

print("number of numerical ens", M)
print("number of DD ens", N)

sig_b = 0.8
#############################################################################################################


#############################################################################################################
# Memory Allocation
#############################################################################################################
T = 300  # this is days

psi_num = np.zeros([M, 2 * Nlat * Nlon])
psi_ML = np.zeros([N, 2 * Nlat * Nlon])
psi_true = np.zeros([1, 2 * Nlat * Nlon])

psi_updated = np.zeros([T + 1, Nlat, Nlon, 2])

psi_true_average = np.zeros([Nlat, Nlon, 2])

metrics = np.zeros((T + 1, 6))
#############################################################################################################

#############################################################################################################
# Initial condition
#############################################################################################################
for k in range(0, N):
    psi_num[k, :] = psi0.flatten() + np.random.normal(
        0,
        sig_b,
        [
            2 * Nlat * Nlon,
        ],
    )

# the last one is for the exact solution
psi_true[0, :] = psi0.flatten()
print(np.shape(psi_true[0, :]))
A = psi_true[0, :].reshape([Nlat, Nlon, 2])
print(np.shape(A), "***")
#############################################################################################################

#############################################################################################################
# Time Advance
#############################################################################################################
count, t = 0, 0
total_MSE1 = 0
total_MSE2 = 0
while t < T + 1:
    # ========================================================================================================
    specialStep = 0
    # ========================================================================================================
    # one step of the true solution
    psi_lastTimeStep = psi_num.copy()
    psi_true = oneStep_Numerical(psi_true, Nlat, Nlon)
    psi_true_average = psi_true_average + psi_true[0].reshape([Nlat, Nlon, 2])
    # one step of numerical integration of our ensemble
    psi_num = oneStep_Numerical(psi_num, Nlat, Nlon)
    print(np.shape(psi_true), np.shape(psi_num))
    # ========================================================================================================
    if t > 0 and (t + 1) % DA_cycles == 0:
        print(t + 1, "--------------------------------")
        # Machine learning
        psi_ML = oneStep_ML(psi_num, Nlat, Nlon, N * M, mean_, var_, 0.12)
        # Pay attention to N & M here, because we use numerical ensemble I put M & M as input
        psi_num = EnKF(
            psi_ML, psi_true[0].reshape([Nlat, Nlon, 2]), psi_num, N, M, Nlat, Nlon
        )
        specialStep = 1
    # ========================================================================================================
    # Averaging and metrics
    psi_updated[t, :, :, :] = (np.mean(psi_num, 0)).reshape([Nlat, Nlon, 2])
    res = calcMetrics(
        psi_updated[t, :, :, :],
        psi_true[0].reshape([Nlat, Nlon, 2]),
        psi_true_average / (t + 1),
    )
    metrics[t, :] = res
    if t > 0 and (t) % DA_cycles == 0:
        plotResults(
            Lx,
            Ly,
            psi_updated[t, :, :, :],
            psi_true[0].reshape([Nlat, Nlon, 2]),
            metrics,
            t,
            1,
        )
    else:
        plotResults(
            Lx,
            Ly,
            psi_updated[t, :, :, :],
            psi_true[0].reshape([Nlat, Nlon, 2]),
            metrics,
            t,
            0,
        )

    # ========================================================================================================
    MSE1 = metrics[t, 4]
    MSE2 = metrics[t, 5]
    total_MSE1 += MSE1
    total_MSE2 += MSE2
    t = t + 1
    if specialStep == 0:
        print("Day ", t, " of " + str(T + 1), "    ", res)
    else:
        print("Day ", t, " of " + str(T + 1), " DA ", res)
#############################################################################################################
# Time Advance
#############################################################################################################
# Calculate the mean of MSE1 values
mean_MSE1 = total_MSE1 / (T + 1)
mean_MSE2 = total_MSE2 / (T + 1)

# Print the mean MSE1
print(f"Mean MSE1 over {T + 1} timesteps: {mean_MSE1}")
print(f"Mean MSE2 over {T + 1} timesteps: {mean_MSE2}")
#############################################################################################################