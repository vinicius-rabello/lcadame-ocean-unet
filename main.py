import numpy as np
from KalmanFilters.EnKalman import EnKF  # Import Ensemble Kalman Filter
from DA.DAuNet import (
    oneStep_Numerical,
    oneStep_ML,
)  # Import numerical and ML-based time-stepping functions
from utils.plotResults import plotResults  # Import function to plot results
from utils.calcMetrics import calcMetrics  # Import function to calculate metrics
import constants as const  # Import constants file
import os
import warnings

# Ignore all warnings to avoid cluttering the output
warnings.filterwarnings("ignore")

# Define the directory and file path for saving images
dir_path = r"./images"

# Ensure the directory exists; if not, create it
if not os.path.exists(dir_path):
    try:
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created.")
    except Exception as e:
        print(f"Failed to create directory {dir_path}: {e}")
else:
    print(f"Directory {dir_path} already exists.")

# Define mean and variance for initial conditions or noise
mean_ = const.mean_
var_ = const.var_

#######################################################################
### Load dataset for truth and observations #########

# Define spatial dimensions
Lx = const.Lx  # Size of x (stick to multiples of 10)
Ly = const.Ly  # Size of y

# Load the dataset containing the true state (e.g., ocean currents, weather data)
psi = np.load("data/oneYear.npy")
print(np.shape(psi), "*****")  # Print the shape of the loaded data

# Extract spatial dimensions from the dataset
Nlat = np.size(psi, 1)  # Number of latitude points
Nlon = np.size(psi, 2)  # Number of longitude points

print("Size of Nlat:", Nlat)
print("Size of Nlon:", Nlon)
print("Shape of psi:", psi.shape)

######## Emulate Observation with noise ########

# Define standard deviation for measurement noise
sig_m = const.sig_m

#############################################################################################################
# Prepare Observations from exact simulation
#############################################################################################################
DA_cycles = const.DA_cycles  # Number of time steps between data assimilation cycles
obs = np.zeros(
    [int(np.size(psi, 0) / DA_cycles), Nlat, Nlon, 2]
)  # Initialize observation array

# Extract observations at regular intervals (every DA_cycles time steps)
for i, k in enumerate(range(DA_cycles, np.size(psi, 0), DA_cycles)):
    obs[i, :, :, :] = psi[k, :, :, :]
#############################################################################################################

########### Start initial condition ##########
psi0 = psi[0, :, :, :]  # Initial state for the simulation
##############################################

#############################################################################################################
# Define ensemble sizes
N = const.N  # Number of ML-based ensemble members
M = const.M  # Number of numerical ensemble members

print("Number of numerical ensemble members:", M)
print("Number of ML-based ensemble members:", N)

# Define standard deviation for initial condition noise
sig_b = const.sig_b
#############################################################################################################

#############################################################################################################
# Memory Allocation
#############################################################################################################
T = const.T  # Total simulation time in days

# Initialize arrays for storing numerical and ML-based ensemble states
psi_num = np.zeros([M, 2 * Nlat * Nlon])  # Numerical ensemble states
psi_ML = np.zeros([N, 2 * Nlat * Nlon])  # ML-based ensemble states
psi_true = np.zeros([1, 2 * Nlat * Nlon])  # True state

# Initialize array for storing updated states after data assimilation
psi_updated = np.zeros([T + 1, Nlat, Nlon, 2])

# Initialize array for storing the time-averaged true state
psi_true_average = np.zeros([Nlat, Nlon, 2])

# Initialize array for storing metrics (e.g., RMSE, MSE)
metrics = np.zeros((T + 1, 6))
#############################################################################################################

#############################################################################################################
# Initial condition
#############################################################################################################
# Perturb the initial condition with noise for each ensemble member
for k in range(0, N):
    psi_num[k, :] = psi0.flatten() + np.random.normal(
        0,
        sig_b,
        [
            2 * Nlat * Nlon,
        ],
    )

# Set the true state as the initial condition
psi_true[0, :] = psi0.flatten()
print("Shape of psi_true[0, :]:", np.shape(psi_true[0, :]))

# Reshape the true state for visualization or further processing
A = psi_true[0, :].reshape([Nlat, Nlon, 2])
print("Shape of A:", np.shape(A), "***")
#############################################################################################################

#############################################################################################################
# Time Advance
#############################################################################################################
count, t = 0, 0  # Initialize counters
total_MSE1 = 0  # Accumulator for MSE1
total_MSE2 = 0  # Accumulator for MSE2

while t < T + 1:
    # ========================================================================================================
    specialStep = 0  # Flag to indicate if a data assimilation step is performed
    # ========================================================================================================
    # One step of the true solution (advance the true state in time)
    psi_lastTimeStep = psi_num.copy()
    psi_true = oneStep_Numerical(psi_true, Nlat, Nlon)
    psi_true_average = psi_true_average + psi_true[0].reshape([Nlat, Nlon, 2])

    # One step of numerical integration for the ensemble
    psi_num = oneStep_Numerical(psi_num, Nlat, Nlon)
    print("Shape of psi_true and psi_num:", np.shape(psi_true), np.shape(psi_num))
    # ========================================================================================================

    # Perform data assimilation every DA_cycles time steps
    if t > 0 and (t + 1) % DA_cycles == 0:
        print(t + 1, "--------------------------------")
        # Machine learning-based time step
        psi_ML = oneStep_ML(psi_num, Nlat, Nlon, N * M, mean_, var_, 0.12)
        # Apply Ensemble Kalman Filter to update the ensemble
        psi_num = EnKF(
            psi_ML, psi_true[0].reshape([Nlat, Nlon, 2]), psi_num, N, M, Nlat, Nlon
        )
        specialStep = 1  # Set flag to indicate a data assimilation step
    # ========================================================================================================

    # Averaging and metrics calculation
    psi_updated[t, :, :, :] = (np.mean(psi_num, 0)).reshape([Nlat, Nlon, 2])
    res = calcMetrics(
        psi_updated[t, :, :, :],
        psi_true[0].reshape([Nlat, Nlon, 2]),
        psi_true_average / (t + 1),
    )
    metrics[t, :] = res

    # Plot results at regular intervals
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
    # Accumulate MSE values for averaging later
    MSE1 = metrics[t, 4]
    MSE2 = metrics[t, 5]
    total_MSE1 += MSE1
    total_MSE2 += MSE2

    # Increment time step
    t = t + 1

    # Print progress
    if specialStep == 0:
        print("Day ", t, " of " + str(T + 1), "    ", res)
    else:
        print("Day ", t, " of " + str(T + 1), " DA ", res)
#############################################################################################################

# Calculate the mean of MSE1 and MSE2 values over the simulation
mean_MSE1 = total_MSE1 / (T + 1)
mean_MSE2 = total_MSE2 / (T + 1)

# Print the mean MSE1 and MSE2
print(f"Mean MSE1 over {T + 1} timesteps: {mean_MSE1}")
print(f"Mean MSE2 over {T + 1} timesteps: {mean_MSE2}")
#############################################################################################################
