"""Simulate images on the Rubin focal plane from a catalog of stars."""
from typing import Dict, Union

import batoid
import galsim
import numpy as np
import numpy.typing as npt
import wfsim
from astropy import table


class ImageSimulator:
    """Simulate images on the Rubin focal plane from a catalog of stars."""

    rafts = ["R00", "R40", "R04", "R44"]
    extra_chips = [f"{raft}_SW0" for raft in rafts]
    intra_chips = [f"{raft}_SW1" for raft in rafts]

    # area of the Rubin primary mirror in cm^2
    m1Area = np.pi * 418**2 * (1 - 0.61**2)

    def __init__(
        self,
        dof: npt.NDArray[np.float64],
        filterName: str,
        zenith: float,
        raw_seeing: float,
        expTime: float,
        temperature: float,
        pressure: float,
        H2O_pressure: float,
        screen_size: float,
        screen_scale: float,
        nproc: int,
        rng: np.random.Generator,
    ) -> None:
        """
        Parameters
        ----------
        dof: np.ndarray
            The degrees of freedom that perturb the telescope and mirror.
        filterName: str
            One of the LSST filters: u, g, r, i, z, y.
        zenith: galsim.Angle
            The zenith angle of the pointing. Sets the airmass.
        raw_seeing: galsim.Angle
            The zenith 500nm seeing in arcseconds.
        expTime: float
            The exposure time in seconds.
        temperature: float
            The temperature in Kelvin.
        pressure: float
            The air pressure in kPa.
        H2O_pressure: float
            The pressure of water in the air in kPa.
        screen_size: float
        screen_scale: float
        nproc: int
            The number of CPUs used to create screens in parallel.
        rng: np.random.Generator
            A numpy random generator.
        """
        # save the dofs, observation, and atmosphere params
        self._dof = dof
        self._obs_kwargs = {
            "zenith": zenith * galsim.degrees,
            "raw_seeing": raw_seeing * galsim.arcsec,
            "exptime": expTime,
            "temperature": temperature,
            "pressure": pressure,
            "H2O_pressure": H2O_pressure,
        }
        self._atm_kwargs = {
            "screen_size": screen_size,
            "screen_scale": screen_scale,
            "nproc": nproc,
        }

        # load the bandpass
        self._bandpass = galsim.Bandpass(
            f"LSST_{filterName}.dat", wave_type="nm"
        ).withZeropoint("AB")
        self._obs_kwargs["wavelength"] = self._bandpass.effective_wavelength

        # load the telescope
        telescope = batoid.Optic.fromYaml(f"LSST_{filterName}.yaml")

        # perturb the telescope
        factory = wfsim.SSTFactory(telescope)
        self._telescope = factory.get_telescope(dof=dof)

        # create the simulator
        self._simulator = wfsim.SimpleSimulator(
            self._obs_kwargs,
            self._atm_kwargs,
            self._telescope,
            self._bandpass,
            rng=rng,
        )

    @property
    def dof(self) -> npt.NDArray[np.float64]:
        """Return the telescope perturbations."""
        return self._dof

    @property
    def bandpass(self) -> galsim.Bandpass:
        """Return the bandpass."""
        return self._bandpass

    @property
    def obs_kwargs(self) -> Dict[str, float]:
        """Return the observation params."""
        return self._obs_kwargs

    @property
    def atm_kwargs(self) -> Dict[str, Union[float, int]]:
        """Return the atmosphere params."""
        return self._atm_kwargs

    def getSimulator(self, chip: str) -> wfsim.SimpleSimulator:
        """Return the simulator for the corresponding chip.

        Parameters
        ----------
        chip: str
            The name of the chip you want the simulator for.

        Returns
        -------
        wfsim.SimpleSimulator
            The wfsim simulator for the chip requested.
        """
        # load the current simulator and telescope
        simulator = self._simulator
        telescope = self._telescope

        # set the focal distance of the telescope
        if chip in self.extra_chips:
            simulator.telescope = telescope.withGloballyShiftedOptic(
                "Detector", [0, 0, +0.0015]
            )
        elif chip in self.intra_chips:
            simulator.telescope = telescope.withGloballyShiftedOptic(
                "Detector", [0, 0, -0.0015]
            )
        else:
            simulator.telescope = telescope.withGloballyShiftedOptic(
                "Detector", [0, 0, 0]
            )

        # set the name of the simulator to the chip name
        # this sets the location, size, and orientation of the chip
        simulator.set_name(chip)

        return simulator

    def _correctCentroids(self, catalog: table.Table) -> table.Table:
        """Correct the centroids so they correspond to donut centers from batoid.

        I believe this is because LSST uses a tangent projection which distorts
        positions far from the center of the focal plane.

        Parameters
        ----------
        catalog: astropy.table.Table
            The catalog of stars whose centroids need correction.

        Returns
        -------
        astropy.table.Table
            The catalog with corrected centroids.
        """
        # don't modify original
        catalog = catalog.copy()

        for detector in set(catalog["detector"]):
            # get the simulator for this detector
            simulator = self.getSimulator(detector)

            # convert the field angles to pixels
            xCentroid, yCentroid = simulator.wcs.radecToxy(
                catalog[catalog["detector"] == detector]["xField"],
                catalog[catalog["detector"] == detector]["yField"],
                galsim.radians,
            )

            # shift coordinate system to bottom left of image
            xCentroid -= simulator.image.bounds.xmin
            yCentroid -= simulator.image.bounds.ymin

            # convert these pixels to pixels in the LSST coordinate system
            # https://github.com/jmeyers314/wfsim/blob/main/notebooks/
            #     Focal%20Plane%20Coordinate%20Systems.ipynb
            if detector in ["R00_SW0", "R44_SW1"]:
                xCentroid, yCentroid = 4072 - xCentroid, 2000 - yCentroid
            elif detector in ["R40_SW0", "R04_SW1"]:
                xCentroid, yCentroid = yCentroid, 2000 - xCentroid
            elif detector in ["R40_SW1", "R04_SW0"]:
                xCentroid, yCentroid = 4072 - yCentroid, xCentroid
            elif detector in ["R00_SW1", "R44_SW0"]:
                pass

            # save the new centroids
            catalog["xCentroid"][catalog["detector"] == detector] = xCentroid
            catalog["yCentroid"][catalog["detector"] == detector] = yCentroid

        return catalog

    def cutoutStamps(
        self,
        images: Dict[str, npt.NDArray[np.float64]],
        catalog: table.Table,
        cropRadius: int = 80,
    ) -> Dict[str, npt.NDArray[np.float64]]:
        """Cutout the individual donut stamps.

        Parameters
        ----------
        images: dict
            Dictionary of CWFS images.
        catalog: astropy.table.Table
            The catalog of stars in the CWFS images.

        Returns
        -------
        dict
            A dictionary of donut postage stamps.
        """

        # get the corrected centroids
        catalog = self._correctCentroids(catalog)

        stamps = dict()
        for cat in catalog.group_by("detector").groups:
            # get the detector name
            detector = cat["detector"][0]

            # get the AOS sources
            if "aosSource" in cat.columns:
                cat = cat[cat["aosSource"]]

            # if the catalog is empty, move on
            if len(cat) == 0:
                continue

            # get the centroid pixels of the donuts
            xCentroid = cat["xCentroid"].value.round().astype(int)
            yCentroid = cat["yCentroid"].value.round().astype(int)

            # convert the LSST pixels coords into the numpy pixel coords
            # note this assumes that you are NOT simulating on guide sensors
            if detector in ["R00_SW0", "R44_SW1"]:
                xCentroid, yCentroid = 4072 - xCentroid, 2000 - yCentroid
            elif detector in ["R40_SW0", "R04_SW1"]:
                xCentroid, yCentroid = 2000 - yCentroid, xCentroid
            elif detector in ["R40_SW1", "R04_SW0"]:
                xCentroid, yCentroid = yCentroid, 4072 - xCentroid
            elif detector in ["R00_SW1", "R44_SW0"]:
                pass

            # get the image for this detector
            img = images[detector]

            # crop the donuts
            for ID, x, y in zip(cat["objectId"], xCentroid, yCentroid):
                cutout = img[
                    max(y - cropRadius, 0) : min(y + cropRadius, img.shape[0]),
                    max(x - cropRadius, 0) : min(x + cropRadius, img.shape[1]),
                ].copy()
                # for some reason, the BBox cut doesn't catch all of the donuts
                # that are really near the edge? Anyway, for now I have this hack
                # to skip donuts whose cutouts aren't square
                if cutout.shape[0] != cutout.shape[1]:
                    continue
                stamps[ID] = cutout

        return stamps

    def simulateCatalog(
        self,
        catalog: table.Table,
        rng: np.random.Generator,
        background: bool = None,
        returnStamps: bool = True,
        cropRadius: int = 85,
    ) -> Dict[str, npt.NDArray[np.float64]]:
        """Simulate CWFS images for the given catalog.

        Parameters
        ----------
        catalog: Table
            The catalog of stars to simulate.
        rng: np.random.Generator
            A numpy random generator.
        background: float, optional
            The sky brightness in ABmag / arcsec^2.
            If None, no background is simulated.
        returnStamps: bool, default=True
            Whether to return donut stamps instead of the full CWFS images.
        cropRadius: int, default=85
            The radius (half the width) of the cutout postage stamps.

        Returns
        -------
        dict
            Dictionary of the form {chip_name: simulated_image}
        """
        # get the list of detectors
        detectors = set(catalog["detector"])

        # if we are simulating the sky background, get the photon flux
        if background is not None:
            skySed = galsim.SED(lambda x: 1, "nm", "flambda").withMagnitude(
                background, self.bandpass
            )
            skyFlux = skySed.calculateFlux(
                self.bandpass
            )  # photons / cm^2 / s / arcsec^2
            nSkyPhotons = (
                skyFlux * self._obs_kwargs["exptime"] * self.m1Area
            )  # photons / arcsec^2

        # loop through the detectors
        images = {}
        for detector in detectors:
            # get the simulator for this detector
            simulator = self.getSimulator(detector)

            # simulate stars on this detector
            for star in catalog[catalog["detector"] == detector]:
                sed = wfsim.BBSED(star["temperature"]).withMagnitude(
                    star["lsstMag"], self.bandpass
                )
                nPhotons = int(
                    sed.calculateFlux(self.bandpass)
                    * self._obs_kwargs["exptime"]
                    * self.m1Area
                )
                simulator.add_star(star["xField"], star["yField"], sed, nPhotons, rng)

            # add background
            if background is not None:
                sqArcsecPerPixel = (
                    np.prod(np.diff(simulator.get_bounds(galsim.arcsec), axis=1))
                    / simulator.image.array.size
                )
                simulator.add_background(nSkyPhotons * sqArcsecPerPixel, rng)

            # save the detector image in the dictionary
            images[detector] = simulator.image.array.copy()

        # cutout the individual donut stamps
        if returnStamps:
            images = self.cutoutStamps(images, catalog, cropRadius)

        return images
