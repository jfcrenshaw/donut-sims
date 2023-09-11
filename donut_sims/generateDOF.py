"""Generate random degrees of freedom for telescope perturbations."""
import numpy as np
from scipy.special import erf, gamma


def generateDOF(
    rng: np.random.Generator, norm: float = 0.25, size: int = 1
) -> np.ndarray:
    """Generate random perturbations to the telescope.

    The degrees of freedom (DOFs) are drawn from a multidimensional Laplace
    distribution. The scale parameter for each DOF is such that the expected
    contribution to the PSF is the same for each DOF. The expected contribution
    from all DOFs is equal to `norm`.

    Parameters
    ----------
    rng: np.random.Generator
        A numpy random generator.
    norm: float, default = 0.25
        The expected PSF contribution from all 50 DOFs.
    size: int, default=1
        How many samples to draw.

    Returns
    -------
    np.ndarray
        Numpy array of sampled degrees of freedom.
    """
    # scales for each DOF
    # units are arcsec / unit of perturbation
    dof_scale = np.array(
        [
            2.64e-02,  # 0
            1.08e-03,
            1.08e-03,
            3.28e-02,
            3.28e-02,
            2.64e-02,  # 5
            2.86e-04,
            2.86e-04,
            2.51e-02,
            2.51e-02,
            9.83e-01,  # 10
            9.81e-01,
            6.97e00,
            1.40e00,
            1.35e00,
            6.26e00,  # 15
            6.24e00,
            1.67e00,
            2.55e00,
            3.15e00,
            3.27e00,  # 20
            3.32e00,
            4.07e00,
            2.53e00,
            4.06e00,
            1.40e01,  # 25
            5.00e00,
            7.92e00,
            3.03e00,
            1.90e01,
            6.51e-01,  # 30
            6.51e-01,
            9.58e-01,
            9.88e-01,
            3.31e00,
            3.24e00,  # 35
            3.24e00,
            1.57e00,
            1.28e00,
            2.82e00,
            3.21e00,  # 40
            1.74e00,
            1.76e00,
            3.06e00,
            3.17e00,
            1.78e00,  # 45
            1.75e00,
            1.06e01,
            1.19e01,
            1.20e01,  # 49
        ]
    )

    # normal scale
    # i.e. this is the expected norm for a Normal vector
    ndof = len(dof_scale)
    normal_scale = np.sqrt(2) * gamma((ndof + 1) / 2) / gamma(ndof / 2)

    dofs = rng.normal(
        scale=1 / dof_scale / normal_scale,
        size=(size, ndof),
    )

    # draw a total norm from the folded Gaussian distribution
    # the folded Gaussian has mu=sigma, and mu chosen such that the expected
    # value of rand_norm = norm
    mu = norm / (np.sqrt(2 / np.pi) * np.exp(-1 / 2) + erf(1 / np.sqrt(2)))
    rand_norm = np.abs(rng.normal(mu, mu, size=(size, 1)))

    # rescale so that the total norms match
    dofs *= rand_norm

    return dofs.squeeze()
