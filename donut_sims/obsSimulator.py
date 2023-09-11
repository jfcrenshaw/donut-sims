"""Simulator that ties together Observation Schedule, Gaia Sources, and Image sim."""
from typing import Dict, List, Union

import numpy as np
from astropy import table

from donut_sims import GaiaSourceSelector, ImageSimulator, ObsScheduler


class ObsSimulator:
    """Simulate Observations on the corner wavefront sensors."""

    def __init__(
        self,
        opsimPath: str = "/astro/store/epyc/users/jfc20/rubin_sim_data/sim_baseline/baseline.db",
        butlerRepo: str = "/epyc/data/lsst_refcats/gen3",
        collections: List[str] = ["refcats"],
        refCatName: str = "gaia_dr2_20200414",
        donutRadius: float = 75,
        checkDir: str = None,
    ) -> None:
        """
        Parameters
        ----------
        opsimPath: str
            Path to the OpSim simulation database.
        butlerRepo: str, default="/epyc/data/lsst_refcats/gen3"
            Path to the butler repository.
        collections: list, default=["refcats"]
            The collections to be searched when loading the data.
        refCatName: str, default="gaia_dr2_20200414"
            Name of the reference catalog.
        donutRadius: float, default=75
            The donut radius used to determine whether donuts are blended.
        checkDir: str, optional
            The directory to check for pre-existing simulations.
            Any already-simulated pointings will be skipped.
        """
        # create the Observation Scheduler
        self.obsScheduler = ObsScheduler(opsimPath=opsimPath, checkDir=checkDir)

        # create the Source Selector
        self.sourceSelector = GaiaSourceSelector(
            butlerRepo=butlerRepo,
            collections=collections,
            refCatName=refCatName,
            donutRadius=donutRadius,
        )

        # no cached observation or simulator at instantiation
        self._cache_obs = None
        self._cache_simulator = None

    def simulateObs(
        self,
        dof: np.ndarray,
        rng: np.random.Generator,
        recomputeAtm: bool = True,
        maxAosSources: int = 4,
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
    ) -> Dict[str, Union[table.Row, table.Table, np.ndarray]]:
        """
        Parameters
        ----------
        dof: np.ndarray
            The degrees of freedom that perturb the telescope and mirror.
        rng: np.random.Generator
            A numpy random number generator.
        recomputeAtm: bool, default=True
            Whether to recompute the atmosphere using the new OpSim selection.
            If False, the new OpSim selection is only used to generate a new
            star catalog at a new pointing.
        maxAosSources: int, default=3
            The maximum number of AOS sources per detector.
        expTime: int, default=15
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
                new_obs = self.obsScheduler.getRandomObservation(rng)

                # determine the filter we are observing in
                lsstFilter = (
                    new_obs["lsstFilter"]
                    if recomputeAtm
                    else self._cache_obs["lsstFilter"]  # type: ignore
                )

                # get a catalog from the pointing
                catalog = self.sourceSelector.selectSources(
                    boresightRA=new_obs["boresightRa"],
                    boresightDec=new_obs["boresightDec"],
                    boresightRotAng=new_obs["boresightRotAng"],
                    lsstFilter=lsstFilter,
                    selectAosSources=True,
                    maxAosSources=maxAosSources,
                    rng=rng,
                )
            except:
                pass

        # if we are recomputing the atmosphere, we must re-build the image simulator
        if recomputeAtm:
            print("building the image simulator")
            simulator = ImageSimulator(
                dof=dof,
                filterName=new_obs["lsstFilter"],
                zenith=np.arccos(1 / new_obs["airmass"]),
                raw_seeing=new_obs["seeingFwhm500"],
                expTime=expTime,
                temperature=temperature,
                pressure=pressure,
                H2O_pressure=H2O_pressure,
                screen_size=screen_size,
                screen_scale=screen_scale,
                nproc=nproc,
                rng=rng,
            )
            self._cache_simulator = simulator

            # get the sky background
            background = new_obs["skyBrightness"] if background else None

            # and cache the new observation
            self._cache_obs = new_obs

        else:
            # use the cached simulator
            simulator = self._cache_simulator

            # and the corresponding sky background
            background = self._cache_obs["skyBrightness"] if background else None

            # but set the new dof
            simulator.setDOF(dof)

        # simulate the images
        print("simulating the images")
        images = simulator.simulateCatalog(
            catalog, rng, background, returnStamps, cropRadius
        )

        # correct the centroids
        print("correcting centroids")
        catalog = simulator._correctCentroids(catalog)

        # add pointing IDs to match pointings to the opsim ID
        catalog.add_column(len(catalog) * [new_obs["observationId"]], 0, "pointingId")

        # add observation IDs to match observing conditions to the opsim ID
        catalog.add_column(
            len(catalog) * [self._cache_obs["observationId"]], 1, "observationId"
        )

        return {
            "catalog": catalog,
            "images": images,
        }
