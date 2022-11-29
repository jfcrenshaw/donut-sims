"""Calculate zernikes from telescope degrees of freedom."""
import batoid
import galsim
import numpy as np
import numpy.typing as npt
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
    dof: npt.NDArray[np.float64],
    detector: str,
    band: str = "r",
) -> npt.NDArray[np.float64]:
    """Calculate zernikes for the detector given the perturbations.

    Note the zernikes are calculated at the center of the CWFS.
    For example, if detector == "R00_SW0", then the zernikes
    are calculated at the center of R00_SW0 U R00_SW1.

    The Noll indices of zernikes returned are 4-22.

    Parameters
    ----------
    dof: np.ndarray
        The degrees of freedom used to perturb the telescope.
    detector: str
        The name of the detector to calculate zernikes for.
    band: str, default="r"
        The name of the band the images are observed in.

    Returns
    -------
    np.ndarray
        The array of zernikes for the detector, with Noll indixes 4-22.
        (units: microns)
    """
    # get the location where we calculate zernikes
    location = detectorLocations[detector]

    # perturb the telescope
    bandpass = galsim.Bandpass(f"LSST_{band}.dat", wave_type="nm")
    telescope = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
    factory = wfsim.SSTFactory(telescope)
    perturbed_telescope = factory.get_telescope(dof=dof)

    # calculate the zernikes
    zernikes = batoid.zernike(
        perturbed_telescope,
        location[0],
        location[1],
        bandpass.effective_wavelength * 1e-9,  # nm -> m
        jmax=23,
    )

    # convert from waves -> microns
    zernikes *= bandpass.effective_wavelength / 1e3

    return zernikes[4:]
