import numpy as np

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
