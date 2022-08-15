from pathlib import Path
from typing import Dict, Union

import batoid
import galsim
import lsst.afw.image as afwImage
import lsst.daf.butler as dafButler
import lsst.geom
import numpy as np
import numpy.typing as npt
import wfsim
from astropy import table
from lsst.ts.wep.task.DonutStamp import DonutStamp
from lsst.ts.wep.task.DonutStamps import DonutStamps
from lsst.ts.wep.Utility import DefocalType, getModulePath


class SimsToButler:
    """Save simulations in the butler.

    The butler where the simulations are saved is the gen3TestRepo in ts_wep:
    ts_wep/tests/testData/gen3TestRepo/mlStamps
    """

    # these detectors are also numbered by Rubin
    detectorNumbers = {
        "R00_SW0": 191,
        "R00_SW1": 190,
        "R40_SW0": 199,
        "R40_SW1": 200,
        "R04_SW0": 195,
        "R04_SW1": 194,
        "R44_SW0": 203,
        "R44_SW1": 204,
    }

    # the field angles in radians of the center of the CWFSs
    # the centers are the centers of the unions of the extra
    # and intra focal chips
    detectorLocations = {
        "R00_SW0": (-0.02075, -0.02075),
        "R00_SW1": (-0.02075, -0.02075),
        "R40_SW0": (-0.02075, +0.02075),
        "R40_SW1": (-0.02075, +0.02075),
        "R04_SW0": (+0.02075, -0.02075),
        "R04_SW1": (+0.02075, -0.02075),
        "R44_SW0": (+0.02075, +0.02075),
        "R44_SW1": (+0.02075, +0.02075),
    }

    def __init__(self) -> None:
        wepExampleRepo = Path(getModulePath()) / "tests" / "testData" / "gen3TestRepo"
        self._butler = dafButler.Butler(wepExampleRepo, writeable=True)
        self.butler.registry.registerRun("mlStamps")

    @property
    def butler(self) -> dafButler.Butler:
        """Return the Butler."""
        return self._butler

    @property
    def registry(self) -> dafButler.Registry:
        """Return the Registry."""
        return self.butler.registry

    def wrapStamps(
        self,
        stamps: Dict[str, npt.NDArray[np.float64]],
        catalog: table.Table,
    ) -> Dict[str, DonutStamps]:
        """Wrap the donut stamps in the DonutStamps class.

        Parameters
        ----------
        stamps: dict
            The dictionary of postage stamps.
        catalog: astropy.table.Table
            Astropy table of the catalog corresponding to the postage stamps.

        Returns
        -------
        dict
            A dictionary of DonutStamps objects.
        """
        sources = catalog[catalog["aosSource"]]

        donutStamps: Dict[str, DonutStamps] = {}
        for cat in sources.group_by("detector").groups:
            # get the detector name
            detector = cat["detector"][0]

            # get the defocal type
            if detector[-1] == "0":
                defocalType = DefocalType.Extra
            elif detector[-1] == "1":
                defocalType = DefocalType.Intra

            donutStamps[detector] = []
            for star in cat:

                # some aosSources aren't in the stamps dict because their stamps
                # weren't square for some reason. For now, I am skipping these
                # I will probably want to fix this later
                if star["objectId"] not in stamps:
                    continue

                # put the stamp in an afw masked image
                maskedImage = afwImage.MaskedImageF(
                    image=afwImage.ImageF(stamps[star["objectId"]])
                )

                # save the stamp as a DonutStamp
                donutStamps[detector].append(
                    DonutStamp(
                        maskedImage,
                        lsst.geom.SpherePoint(
                            star["ra"], star["dec"], lsst.geom.radians
                        ),
                        lsst.geom.Point2D(star["xCentroid"], star["yCentroid"]),
                        defocalType,
                        detector,
                        "LSSTCam",
                    )
                )

            # wrap the list in the DonutStamps class
            donutStamps[detector] = DonutStamps(donutStamps[detector])

        return donutStamps

    def calculateZernikes(
        self,
        detector: str,
        dof: npt.NDArray[np.float64],
        band: str = "r",
    ) -> npt.NDArray[np.float64]:
        """Calculate zernikes for the detector given the perturbations.

        Note the zernikes are calculated at the center of the CWFS.
        So for example, if detector == "R00_SW0", then the zernikes
        are calculated at the center of R00_SW0 U R00_SW1.

        The Noll indices of zernikes returned are 4-22.

        Parameters
        ----------
        detector: str
            The name of the detector to calculate zernikes for.
        dof: np.ndarray
            The degrees of freedom used to perturb the telescope.
        band: str, default="r"
            The name of the band the images are observed in.

        Returns
        -------
        np.ndarray
            The array of zernikes for the detector, with Noll indixes 4-22.
        """
        # get the location where we calculate zernikes
        location = self.detectorLocations[detector]

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
            jmax=22,
        )

        return zernikes

    def saveSimulation(
        self,
        name: str,
        observation: Dict[str, Union[npt.NDArray[np.float64], table.Table, table.Row]],
        dof: npt.NDArray[np.float64],
    ) -> None:
        """Save the donut simulations and the corresponding zernikes in the butler.

        Parameters
        ----------
        name: str
            The name of the visit to save in the butler.
        observation: dict
            The dictionary for the observation. Returned from ObsSimulator.simulateObs.
        dof: np.ndarray
            The degrees of freedom that were used to perturb the telescope.
        """
        # pull out stamps, catalog, and metadata from the observation
        stamps = observation["images"]
        catalog = observation["catalog"]
        metadata = observation["metadata"]

        # get the observation ID
        observationId = int(metadata["observationId"])

        # and bandpass
        band = metadata["lsstFilter"]

        # save the metadata
        self.registry.insertDimensionData(
            "visit",
            {
                "name": name,
                "visit": observationId,
                "instrument": "LSSTCam",
                "physical_filter": band,
            },
        )

        # wrap stamps in the DonutStamps object
        donutStamps = self.wrapStamps(stamps, catalog)

        # loop through the detectors
        for detector, dS in donutStamps.items():

            # get the dataset type
            if detector[-1] == "0":
                donutStampsDatasetType = self.registry.getDatasetType(
                    "donutStampsExtra"
                )
            if detector[-1] == "1":
                donutStampsDatasetType = self.registry.getDatasetType(
                    "donutStampsIntra"
                )

            # create the data reference
            dRef = dafButler.DatasetRef(
                donutStampsDatasetType,
                dataId={
                    "visit": observationId,
                    "detector": self.detectorNumbers[detector],
                    "instrument": "LSSTCam",
                },
            )

            # save the stamps in the butler
            self.butler.put(dS, dRef, run="mlStamps")

            # calculate zernikes
            zernikes = self.calculateZernikes(detector, dof, band)

            # create the data reference
            dRef = dafButler.DatasetRef(
                self.registry.getDatasetType("zernikeEstimateAvg"),
                dataId={
                    "visit": observationId,
                    "detector": self.detectorNumbers[detector],
                    "instrument": "LSSTCam",
                },
            )

            # save the zernikes in the butler
            self.butler.put(zernikes, dRef, run="mlStamps")
