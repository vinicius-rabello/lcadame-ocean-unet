import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from keras.layers import (
#     Input,
#     Conv2D,
#     MaxPooling2D,
#     UpSampling2D,
#     ZeroPadding2D,
#     Concatenate,
#     Cropping2D
# )

from keras.callbacks import History


# history = History()

# from keras.models import Model
from KalmanFilters.EnKalman import EnKF
from models.stn import model
import os
from PIL import Image

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

#####################################################################################
# from tensorflow.keras.optimizers import Adam

from scipy.sparse.linalg import cg, LinearOperator

# Define the directory and file path
dir_path = r"./images"
file_path = os.path.join(dir_path, "Solver_000000.png")

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

# Check if the file exists
if os.path.isfile(file_path):
    print(f"File {file_path} exists. Deleting the file...")
    try:
        # Delete the file
        os.remove(file_path)
        print("File deleted.")
    except Exception as e:
        print(f"Failed to delete file {file_path}: {e}")

# Generate a new image file
try:
    print("Generating the file again...")
    image = Image.new("RGB", (100, 100), color=(73, 109, 137))
    image.save(file_path)
    print(f"File {file_path} generated successfully.")
except Exception as e:
    print(f"Failed to generate the file {file_path}: {e}")


### Define Data-driven architecture ######
# def stn(input_shape):
#     inputs = Input(shape=input_shape)

#     padd = ZeroPadding2D(((1, 1), (0, 0)))(inputs)

#     layer1_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(padd)
#     layer2_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer1_conv
#     )
#     layer3_pool = MaxPooling2D(pool_size=(2, 2))(layer2_conv)

#     layer4_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer3_pool
#     )
#     layer5_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer4_conv
#     )
#     layer6_pool = MaxPooling2D(pool_size=(2, 2))(layer5_conv)

#     layer7_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer6_pool
#     )
#     layer8_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer7_conv
#     )

#     layer10_up = Concatenate(axis=-1)(
#         [
#             Conv2D(32, (2, 2), activation="relu", padding="same")(
#                 UpSampling2D(size=(2, 2))(layer8_conv)
#             ),
#             layer5_conv,
#         ]
#     )
#     layer11_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer10_up
#     )
#     layer12_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer11_conv
#     )

#     layer14_up = Concatenate(axis=-1)(
#         [
#             Conv2D(32, (2, 2), activation="relu", padding="same")(
#                 UpSampling2D(size=(2, 2))(layer12_conv)
#             ),
#             layer2_conv,
#         ]
#     )
#     layer15_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer14_up
#     )
#     layer16_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
#         layer15_conv
#     )

#     cropped_outputs = Cropping2D(((1, 1), (0, 0)))(layer16_conv)
#     outputs = Conv2D(2, (5, 5), activation="linear", padding="same")(cropped_outputs)

#     model = Model(inputs, outputs)

#     return model




# model = stn(input_shape=(46, 68, 2))
# model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["accuracy"])
# model.summary()
model.load_weights("G_46_68.weights.h5")
mean_ = np.array([0.00025023, -0.00024681])
var_ = np.array([9.9323115, 0.18261143])
#####################################################################################


########################################################################################
# N is in x & N2 is in Y
# psiIN must be in lat,lon,2
########################################################################################
def drymodel(psiIN, finalT):
    opt = 3  # 1 = just the linear parts, 2 = just the nonlinear parts, 3 = full model

    N = np.size(psiIN, axis=0)  # zonal size of spectral decomposition
    N2 = np.size(psiIN, axis=1)  # meridional size of spectral decomposition
    Lx = 46.0  # size of x -- stick to multiples of 10
    Ly = 68.0  # size of y -- stick to multiples of 10

    nu = 2.3 * pow(10.0, -6.0)  # viscous dissipation
    tau_d = 100.0  # Newtonian relaxation time-scale for interface
    tau_f = 15.0  # surface friction
    beta = 0.196  # beta
    sigma = 3.5
    U_1 = 1.0

    g = 0.04  # leapfrog filter coefficient

    y = np.linspace(-Ly / 2, Ly / 2, N2)

    # Wavenumbers:
    kk = np.fft.rfftfreq(N, Lx / float(N) / 2.0 / np.pi)  # zonal wavenumbers
    ll = np.fft.fftfreq(N2, Ly / float(N2) / 2.0 / np.pi)  # meridional wavenumbers

    tot_time = finalT  # 1#750 #Length of run
    dt = 0.025  # Timestep
    ts = int(tot_time / dt)  # Total timesteps
    lim = 0  # int(650 / dt )#int(ts / 10 ) #Start saving
    st = int(1.0 / dt)  # How often to save data

    #######################################################
    #  Declare arrays

    # Spectral arrays, only need 3 time-steps

    psic_1 = np.zeros(((3, N2, int(N / 2 + 1)))).astype(complex)
    psic_2 = np.zeros(((3, N2, int(N / 2 + 1)))).astype(complex)
    qc_1 = np.zeros(((3, N2, int(N / 2 + 1)))).astype(complex)
    qc_2 = np.zeros(((3, N2, int(N / 2 + 1)))).astype(complex)
    vorc_1 = np.zeros(((3, N2, int(N / 2 + 1)))).astype(complex)
    vorc_2 = np.zeros(((3, N2, int(N / 2 + 1)))).astype(complex)

    ##>print('complex spectral array shapes: ' + str(psic_1.shape))

    # Real arrays, only need 3 time-steps
    psi_1 = np.zeros(((3, N2, N)))
    psi_2 = np.zeros(((3, N2, N)))
    q_1 = np.zeros(((3, N2, N)))
    q_2 = np.zeros(((3, N2, N)))

    ##>print('real array shapes: ' + str(psi_1.shape))

    #######################################################
    #  Define equilibrium interface height + sponge

    sponge = np.zeros(N2)
    u_eq = np.zeros(N2)

    for i in range(N2):
        y1 = float(i - N2 / 2) * (y[1] - y[0])
        y2 = float(min(i, N2 - i - 1)) * (y[1] - y[0])
        sponge[i] = U_1 / (np.cosh(abs(y2 / sigma))) ** 2
        u_eq[i] = U_1 * (
            1.0 / (np.cosh(abs(y1 / sigma))) ** 2
            - 1.0 / (np.cosh(abs(y2 / sigma))) ** 2
        )

    psi_Rc = -np.fft.fft(u_eq) / 1.0j / ll
    psi_Rc[0] = 0.0
    psi_R = np.fft.ifft(psi_Rc)

    #######################################################
    #  Spectral functions

    def ptq(ps1, ps2):
        """Calculate PV"""
        q1 = -(ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2) * ps1 - (
            ps1 - ps2
        )  # -(k^2 + l^2) * psi_1 -0.5*(psi_1-psi_2)
        q2 = -(ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2) * ps2 + (
            ps1 - ps2
        )  # -(k^2 + l^2) * psi_2 +0.5*(psi_1-psi_2)
        return q1, q2

    def qtp(q1_s, q2_s):
        """Invert PV"""
        # Checked
        psi_bt = (
            -(q1_s + q2_s) / (ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2) / 2.0
        )  # (psi_1 + psi_2)/2
        psi_bc = (
            -(q1_s - q2_s)
            / (ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2 + 2.0)
            / 2.0
        )  # (psi_1 - psi_2)/2
        psi_bt[0, 0] = 0.0
        psi1 = psi_bt + psi_bc
        psi2 = psi_bt - psi_bc

        return psi1, psi2

    #######################################################
    #  Initial conditions:

    psi1 = np.transpose(psiIN[:, :, 0])
    psi2 = np.transpose(psiIN[:, :, 1])

    psic_1[0] = np.fft.rfft2(psi1)
    psic_2[0] = np.fft.rfft2(psi2)

    # Transfer values:
    psic_1[1, :, :] = psic_1[0, :, :]
    psic_2[1, :, :] = psic_2[0, :, :]

    # Calculate initial PV
    for i in range(2):
        vorc_1[i], vorc_2[i] = ptq(psic_1[i], psic_2[i])
        q_1[i] = np.fft.irfft2(vorc_1[i]) + beta * y[:, np.newaxis]
        q_2[i] = np.fft.irfft2(vorc_2[i]) + beta * y[:, np.newaxis]
        qc_1[i] = np.fft.rfft2(q_1[i])
        qc_2[i] = np.fft.rfft2(q_2[i])

    #######################################################
    # Time-stepping functions

    def calc_nl(psi, qc):
        """ "Calculate non-linear terms, with Orszag 3/2 de-aliasing"""

        N2, N = np.shape(psi)
        ex = int(N * 3 / 2)  # - 1
        ex2 = int(N2 * 3 / 2)  # - 1
        temp1 = np.zeros((ex2, ex)).astype(complex)
        temp2 = np.zeros((ex2, ex)).astype(complex)
        temp4 = np.zeros((N2, N)).astype(complex)  # Final array

        # Pad values:
        temp1[: N2 // 2, :N] = psi[: N2 // 2, :N]
        temp1[ex2 - N2 // 2 :, :N] = psi[N2 // 2 :, :N]

        temp2[: N2 // 2, :N] = qc[: N2 // 2, :N]
        temp2[ex2 - N2 // 2 :, :N] = qc[N2 // 2 :, :N]

        # Fourier transform product, normalize, and filter:
        temp3 = np.fft.rfft2(np.fft.irfft2(temp1) * np.fft.irfft2(temp2)) * 9.0 / 4.0
        temp4[: N2 // 2, :N] = temp3[: N2 // 2, :N]
        temp4[N2 // 2 :, :N] = temp3[ex2 - N2 // 2 :, :N]

        return temp4

    def nlterm(kk, ll, psi, qc):
        """ "Calculate Jacobian"""

        dpsi_dx = 1.0j * kk[np.newaxis, :] * psi
        dpsi_dy = 1.0j * ll[:, np.newaxis] * psi

        dq_dx = 1.0j * kk[np.newaxis, :] * qc
        dq_dy = 1.0j * ll[:, np.newaxis] * qc

        return calc_nl(dpsi_dx, dq_dy) - calc_nl(dpsi_dy, dq_dx)

    def fs(ovar, rhs, det, nu, kk, ll):
        # backward Euler of dq/dt=RHS-aq
        """Forward Step: q^t-1 / ( 1 + 2. dt * nu * (k^4 + l^4 ) ) + RHS"""
        mult = det / (
            1.0 + det * nu * (np.expand_dims(kk, 0) ** 8 + np.expand_dims(ll, 1) ** 8)
        )

        return mult * (ovar / det + rhs)

    def lf(oovar, rhs, det, nu, kk, ll):
        """Leap frog timestepping: q^t-2 / ( 1 + 2. * dt * nu * (k^4 + l^4 ) ) + RHS"""
        mult = (
            2.0
            * det
            / (
                1.0
                + 2.0
                * det
                * nu
                * (np.expand_dims(kk, 0) ** 8 + np.expand_dims(ll, 1) ** 8)
            )
        )
        return mult * (oovar / det / 2.0 + rhs)

    def filt(var, ovar, nvar, g):
        # see Analysis of time filters used with the leapfrog scheme
        """Leapfrog filtering"""
        return var + g * (ovar - 2.0 * var + nvar)

    #######################################################
    #  Main time-stepping loop

    forc1 = np.zeros((N2, N))
    forc2 = np.zeros((N2, N))
    cforc1 = np.zeros((N2, N // 2 + 1)).astype(complex)
    cforc2 = np.zeros((N2, N // 2 + 1)).astype(complex)

    nl1 = np.zeros((N2, N // 2 + 1)).astype(complex)
    nl2 = np.zeros((N2, N // 2 + 1)).astype(complex)

    psiAll = np.zeros(((N, N2, 2)))
    # Timestepping:
    for i in range(1, ts + 1):
        if i % 100 == 0:
            print("Timestep:", i, "/", ts, flush=True)

        if opt > 1:
            # NL terms -J(psi, qc) - beta * v <<### Still we have problem here
            nl1[:, :] = (
                -nlterm(kk, ll, psic_1[1, :, :], vorc_1[1, :, :])
                - beta * 1.0j * kk[np.newaxis, :] * psic_1[1, :, :]
            )
            nl2[:, :] = (
                -nlterm(kk, ll, psic_2[1, :, :], vorc_2[1, :, :])
                - beta * 1.0j * kk[np.newaxis, :] * psic_2[1, :, :]
            )
            # nl1[:, :] = -nlterm( kk, ll, psic_1[1, :, :], vorc_1[1, :, :]) + beta * 1.j * kk[np.newaxis, :] * psic_1[1, :, :]
            # nl2[:, :] = -nlterm( kk, ll, psic_2[1, :, :], vorc_2[1, :, :]) + beta * 1.j * kk[np.newaxis, :] * psic_2[1, :, :]

        if opt != 2:
            # Linear terms
            # Relax interface
            forc1[:, :] = (psi_1[1] - psi_2[1] - psi_R[:, np.newaxis]) / tau_d
            forc2[:, :] = -(psi_1[1] - psi_2[1] - psi_R[:, np.newaxis]) / tau_d

            # Sponge
            forc1[:, :] -= sponge[:, np.newaxis] * (
                q_1[1] - np.mean(q_1[1], axis=1)[:, np.newaxis]
            )
            forc2[:, :] -= sponge[:, np.newaxis] * (
                q_2[1] - np.mean(q_2[1], axis=1)[:, np.newaxis]
            )

            # Convert to spectral space + add friction
            cforc1 = np.fft.rfft2(forc1)
            cforc2 = (
                np.fft.rfft2(forc2)
                + (kk[np.newaxis, :] ** 2 + ll[:, np.newaxis] ** 2) * psic_2[1] / tau_f
            )

        rhs1 = nl1[:] + cforc1[:]
        rhs2 = nl2[:] + cforc2[:]
        # mrhs = mnl[:]

        if i == 1:
            # Forward step
            qc_1[2, :] = fs(qc_1[1, :, :], rhs1[:], dt, nu, kk, ll)
            qc_2[2, :] = fs(qc_2[1, :, :], rhs2[:], dt, nu, kk, ll)
        else:
            # Leapfrog step
            qc_1[2, :, :] = lf(qc_1[0, :, :], rhs1[:], dt, nu, kk, ll)
            qc_2[2, :, :] = lf(qc_2[0, :, :], rhs2[:], dt, nu, kk, ll)

        if i > 1:
            # Leapfrog filter (our target is qc_1[1] which is time n)
            qc_1[1, :] = filt(qc_1[1, :], qc_1[0, :], qc_1[2, :], g)
            qc_2[1, :] = filt(qc_2[1, :], qc_2[0, :], qc_2[2, :], g)

        for j in range(2):
            q_1[j] = np.fft.irfft2(qc_1[j + 1])
            q_2[j] = np.fft.irfft2(qc_2[j + 1])

            # Subtract off beta and invert
            vorc_1[j] = np.fft.rfft2(q_1[j] - beta * y[:, np.newaxis])
            vorc_2[j] = np.fft.rfft2(q_2[j] - beta * y[:, np.newaxis])
            psic_1[j], psic_2[j] = qtp(vorc_1[j], vorc_2[j])
            psi_1[j] = np.fft.irfft2(psic_1[j])
            psi_2[j] = np.fft.irfft2(psic_2[j])

            # Transfer values:
            qc_1[j, :, :] = qc_1[j + 1, :, :]
            qc_2[j, :, :] = qc_2[j + 1, :, :]

        if i > lim:
            if i % st == 0:
                psiAll[:, :, 0] = np.transpose(psi_1[1])
                psiAll[:, :, 1] = np.transpose(psi_2[1])
    return psiAll





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

# %%


########################################################################
def plotResults(Lx, Ly, psi, psi_exact, metrics, index, special):
    Ny = psi.shape[1]

    levels = np.linspace(-2.5, 2.5, 10)
    plt.set_cmap("bwr")
    # Create a figure
    fig = plt.figure(figsize=(12, 6))
    day = int(index / 5)
    fig.suptitle(f"Day {day + 1}", fontsize=16)
    if special == 1:
        fig.patch.set_facecolor("xkcd:mint green")
    else:
        fig.patch.set_facecolor("xkcd:white")

    # Create a main GridSpec layout with 1 row and 3 columns
    # main_gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 0.05,0.07,1], wspace=0.15)
    main_gs = gridspec.GridSpec(2, 3, wspace=0.15)

    x = np.linspace(0, Lx, psi.shape[0])
    y = np.linspace(0, Ly, psi.shape[1])
    X, Y = np.meshgrid(x, y)

    # Plot in the first cell (1st column)
    ax1 = fig.add_subplot(main_gs[0, 0])
    ax1.set_title("Psi_ens 1")
    contour1 = plt.contourf(X, Y, np.transpose(psi[:, :, 0]), levels=levels)

    # Plot in the second cell (2nd column)
    ax4 = fig.add_subplot(main_gs[0, 1])
    ax4.set_title("Psi_exc 1")
    contour2 = plt.contourf(X, Y, np.transpose(psi_exact[:, :, 0]), levels=levels)

    # Plot in the first cell (1st column)
    ax2 = fig.add_subplot(main_gs[1, 0])
    ax2.set_title("Psi_ens 2")
    contour1 = plt.contourf(X, Y, np.transpose(psi[:, :, 1]), levels=levels)

    # Plot in the second cell (2nd column)
    ax5 = fig.add_subplot(main_gs[1, 1])
    ax5.set_title("Psi_exc 2")
    contour2 = plt.contourf(X, Y, np.transpose(psi_exact[:, :, 1]), levels=levels)

    days = np.linspace(0.25, (index + 1) / 5, index + 1)
    # Plot in the second cell (2nd column)
    ax3 = fig.add_subplot(main_gs[0, 2])
    ax3.set_title("Ek")
    # ax3.set_xlim(Ly/2-20,Ly/2+20)
    # ax3.set_ylim(-0.06,0.06)
    ax3.plot(days, metrics[0 : index + 1, 0], "b")
    ax3.plot(days, metrics[0 : index + 1, 1], "r")

    # Plot in the second cell (2nd column)
    ax6 = fig.add_subplot(main_gs[1, 2])
    ax6.set_title("Acc")
    # ax6.set_xlim(Ly/2-20,Ly/2+20)
    # ax6.set_ylim(-0.06,0.06)
    ax6.plot(days, metrics[0 : index + 1, 2], "b")
    ax6.plot(days, metrics[0 : index + 1, 3], "r")

    plt.tight_layout()

    # Add metrics as caption text at the bottom of the figure
    metrics_text = f"Metrics: Ek = {metrics[index, 0]:.3f}, {metrics[index, 1]:.3f}; Acc = {metrics[index, 2]:.3f}, {metrics[index, 3]:.3f}"
    fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=10)

    # Display the plots
    plt.savefig(f"images/Solver_{index:0{6}}.png", dpi=300)
    plt.close()

    return


########################################################################
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


# ======================================================================================


#####################################################################################
def calcMetrics(psi, psi_exact, psi_exact_average):
    a = np.size(psi[0])
    b = np.size(psi[1])
    Ek1 = (
        np.linalg.norm(psi[:, :, 0] - psi_exact[:, :, 0])
        / a
        * np.max(np.absolute(psi_exact[:, :, 0]))
    )
    Ek2 = (
        np.linalg.norm(psi[:, :, 1] - psi_exact[:, :, 1])
        / b
        * np.max(np.absolute(psi_exact[:, :, 1]))
    )

    den11 = np.sum(
        np.multiply(
            psi[:, :, 0] - psi_exact_average[:, :, 0],
            psi[:, :, 0] - psi_exact_average[:, :, 0],
        )
    )
    den12 = np.sum(
        np.multiply(
            psi_exact[:, :, 0] - psi_exact_average[:, :, 0],
            psi_exact[:, :, 0] - psi_exact_average[:, :, 0],
        )
    )
    den21 = np.sum(
        np.multiply(
            psi[:, :, 1] - psi_exact_average[:, :, 1],
            psi[:, :, 1] - psi_exact_average[:, :, 1],
        )
    )
    den22 = np.sum(
        np.multiply(
            psi_exact[:, :, 1] - psi_exact_average[:, :, 1],
            psi_exact[:, :, 1] - psi_exact_average[:, :, 1],
        )
    )

    den11 = max(1.0e-12, den11)
    den12 = max(1.0e-12, den12)
    den21 = max(1.0e-12, den21)
    den22 = max(1.0e-12, den22)

    Acc1 = (
        np.sum(
            np.multiply(
                psi[:, :, 0] - psi_exact_average[:, :, 0],
                psi_exact[:, :, 0] - psi_exact_average[:, :, 0],
            )
        )
        / (den11 * den12) ** 0.5
    )
    Acc2 = (
        np.sum(
            np.multiply(
                psi[:, :, 1] - psi_exact_average[:, :, 1],
                psi_exact[:, :, 1] - psi_exact_average[:, :, 1],
            )
        )
        / (den21 * den22) ** 0.5
    )

    # Calculate MSE for psi[:,:,0] and psi[:,:,1]
    MSE1 = np.mean((psi[:, :, 0] - psi_exact[:, :, 0]) ** 2)
    MSE2 = np.mean((psi[:, :, 1] - psi_exact[:, :, 1]) ** 2)

    return np.array([Ek1, Ek2, Acc1, Acc2, MSE1, MSE2])


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


### Load dataset for truth and Obs #########


Lx = 46  # 96  #46. #size of x -- stick to multiples of 10
Ly = 68  # 192 #68.
psi = np.load("ICs/oneYear.npy")
# psi = np.load('Data-LR/oneYearLR.npy')
print(np.shape(psi), "*****")
Nlat = np.size(psi, 1)  # np.size(psi,2)
Nlon = np.size(psi, 2)  # np.size(psi,3)

print("size of Nlat", Nlat)
print("size of Nlon", Nlon)
print("shape of psi", psi.shape)

######## Emulate Observation with noise ########

sig_m = 0.15  # standard deviation for measurement noise
# R = sig_m**2*np.eye(Nlat*Nlon*2,Nlat*Nlon*2)

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
# B = sig_b**2*np.eye(2*Nlat*Nlon,Nlat*Nlon*2)
# Q = 0.0*np.eye(2*Nlat*Nlon,Nlat*Nlon*2)
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
    # print(psi_true[0]-psi_num[0])
    # ========================================================================================================
    if t < 35 and (t + 1) % DA_cycles == 0:
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
    # print(metrics[t, 4] / (t+1))
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
# MSE
# print()
