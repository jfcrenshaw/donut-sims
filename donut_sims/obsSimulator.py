"""Simulator that ties together Observation Schedule, Gaia Sources, and Image sim."""
from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt
from astropy import table

from donut_sims import GaiaSourceSelector, ImageSimulator, ObsScheduler


class ObsSimulator:
    """Simulate Observations on the corner wavefront sensors."""

    def __init__(
        self,
        opsim_path: str = "/astro/store/epyc/users/jfc20/rubin_sim_data/sim_baseline/baseline.db",
        butlerRepo: str = "/epyc/data/lsst_refcats/gen3",
        collections: List[str] = ["refcats"],
        refCatName: str = "gaia_dr2_20200414",
        donutRadius: float = 75,
    ) -> None:
        """
        Parameters
        ----------
        opsim_path: str
            Path to the OpSim simulation database.
        butlerRepo: str, default="/epyc/data/lsst_refcats/gen3"
            Path to the butler repository.
        collections: list, default=["refcats"]
            The collections to be searched when loading the data.
        refCatName: str, default="gaia_dr2_20200414"
            Name of the reference catalog.
        donutRadius: float, default=75
            The donut radius used to determine whether donuts are blended.
        """
        # create the Observation Scheduler
        self.obsScheduler = ObsScheduler(opsim_path=opsim_path)

        # create the Source Selector
        self.sourceSelector = GaiaSourceSelector(
            butlerRepo=butlerRepo,
            collections=collections,
            refCatName=refCatName,
            donutRadius=donutRadius,
        )

    def simulateObs(
        self,
        dof: npt.NDArray[np.float64],
        rng: np.random.Generator,
        expTime: float = 15,
        temperature: float = 293,
        pressure: float = 69,
        H2O_pressure: float = 1,
        screen_size: float = 819.2,
        screen_scale: float = 0.1,
        nproc: int = 6,
        background: bool = True,
        returnStamps: bool = True,
        cropRadius: int = 85,
    ) -> Dict[str, Union[table.Row, table.Table, npt.NDArray[np.float64]]]:
        """
        Parameters
        ----------
        dof: np.ndarray
            The degrees of freedom that perturb the telescope and mirror.
        rng: np.random.Generator
            A numpy random number generator.
        expTime: int
            The exposure time in seconds.
        temperature: float, default=293
            The temperature in Kelvin.
        pressure: float, default=69
            The air pressure in kPa.
        H2O_pressure: float, default=1
            The pressure of water in the air in kPa.
        screen_size: float, default=819.2
        screen_scale: float, default=0.1
        nproc: int, default=6
            The number of CPUs used to create screens in parallel.
        background: bool, default=True
            Whether to simulate the sky background.
        returnStamps: bool, default=True
            Whether to return donut stamps instead of the full CWFS images.
        cropRadius: int, default=85
            The radius (half the width) of the cutout postage stamps.

        Returns
        -------
        dict
            Dictionary containing metadata and the images.
        """

        # get a random observation and the catalog from the pointing
        # this is in a while loop to handle cases where the random pointing
        # is not within the gaia footprint
        print("selecting sources")
        catalog = None
        while catalog is None:
            try:
                # get a random observation
                observation = self.obsScheduler.getRandomObservation(rng)

                # get a catalog from the pointing
                catalog = self.sourceSelector.selectSources(
                    boresightRA=observation["boresightRa"],
                    boresightDec=observation["boresightDec"],
                    boresightRotAng=observation["boresightRotAng"],
                    lsstFilter=observation["lsstFilter"],
                    selectAosSources=True,
                    rng=rng,
                )
            except:
                pass

        # add observation IDs to match stars to the opsim ID
        catalog.add_column(
            len(catalog) * [observation["observationId"]], 0, "observationId"
        )

        # build the image simulator
        print("building the image simulator")
        simulator = ImageSimulator(
            dof=dof,
            filterName=observation["lsstFilter"],
            zenith=np.arccos(1 / observation["airmass"]),
            raw_seeing=observation["seeingFwhm500"],
            expTime=expTime,
            temperature=temperature,
            pressure=pressure,
            H2O_pressure=H2O_pressure,
            screen_size=screen_size,
            screen_scale=screen_scale,
            nproc=nproc,
            rng=rng,
        )

        # set the sky background
        background = observation["skyBrightness"] if background else None

        # simulate the images
        print("simulating the images")
        images = simulator.simulateCatalog(
            catalog, rng, background, returnStamps, cropRadius
        )

        # correct the centroids
        print("correcting centroids")
        catalog = simulator._correctCentroids(catalog)

        return {
            "metadata": observation,
            "catalog": catalog,
            "images": images,
        }
