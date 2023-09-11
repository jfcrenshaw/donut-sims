"""Calculate zernikes from telescope degrees of freedom."""
import batoid
import numpy as np
import wfsim

# the field angles in radians of the center of the CWFSs
# the centers are the centers of the unions of the extra
# and intra focal chips
detectorLocations = {
    # bottom left
    "R00": (-0.02075, -0.02075),
    "R00_SW0": (-0.02075, -0.02075),
    "R00_SW1": (-0.02075, -0.02075),
    # top left
    "R40": (-0.02075, +0.02075),
    "R40_SW0": (-0.02075, +0.02075),
    "R40_SW1": (-0.02075, +0.02075),
    # bottom right
    "R04": (+0.02075, -0.02075),
    "R04_SW0": (+0.02075, -0.02075),
    "R04_SW1": (+0.02075, -0.02075),
    # top right
    "R44": (+0.02075, +0.02075),
    "R44_SW0": (+0.02075, +0.02075),
    "R44_SW1": (+0.02075, +0.02075),
}


def dofsToZernikes(
    dof: np.ndarray,
    detector: str,
    band: str = "r",
    subtract_zk0: bool = True,
) -> np.ndarray:
    """Calculate zernikes for the detector given the perturbations.

    Note the zernikes are calculated at the center of the CWFS.
    For example, if detector == "R00_SW0", then the zernikes
    are calculated at the center of R00_SW0 U R00_SW1.

    The Noll indices of zernikes returned are 4-22.

    Currently, this function uses a fiducial wavelength of 1 micron.

    Parameters
    ----------
    dof: np.ndarray
        The degrees of freedom used to perturb the telescope.
    detector: str
        The name of the detector to calculate zernikes for.
    band: str, default="r"
        The name of the band the images are observed in.
    subtract_zk0: bool, default=False
        Whether to subtract the intrinsic Zernikes.

    Returns
    -------
    np.ndarray
        The array of zernikes for the detector, with Noll indixes 4-22.
        (units: microns)
    """
    # get the location where we calculate zernikes
    location = detectorLocations[detector]

    # perturb the telescope
    telescope = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
    factory = wfsim.SSTFactory(telescope)
    perturbed_telescope = factory.get_telescope(dof=dof)

    # calculate the perturbed zernikes
    zk = batoid.zernike(
        perturbed_telescope,
        location[0],
        location[1],
        1e-6,  # nm -> m
        jmax=22,
        eps=perturbed_telescope.pupilObscuration,
    )

    if subtract_zk0:
        zk -= batoid.zernike(
            telescope,
            location[0],
            location[1],
            1e-6,  # nm -> m
            jmax=22,
            eps=telescope.pupilObscuration,
        )

    return zk[4:]
