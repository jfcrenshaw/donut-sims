"""Class to select observations from OpSim."""
import sqlite3

import astropy.units as u
import numpy as np
import pandas as pd
from astropy import table
from glob import glob


class ObsScheduler:
    """Select observations from the simulated observation scheduler."""

    _columns = {
        "observationID": "observationID",
        "fieldRA": "boresightRa",
        "fieldDec": "boresightDec",
        "rotTelPos": "boresightRotAng",
        "filter": "lsstFilter",
        "airmass": "airmass",
        "seeingFwhm500": "seeingFwhm500",
        "skyBrightness": "skyBrightness",
    }

    _units = {
        "boresightRa": u.deg,
        "boresightDec": u.deg,
        "boresightRotAng": u.deg,
        "seeingFwhm500": u.arcsec,
        "skyBrightness": u.mag("AB / arcsec**2"),
    }

    def __init__(
        self,
        opsimPath: str = "/astro/store/epyc/users/jfc20/rubin_sim_data/sim_baseline/baseline.db",
        checkDir: str = None,
    ) -> None:
        """
        Parameters
        ----------
        opsim_path: str
            Path to the OpSim simulation database.
        checkDir: str, optional
            The directory to check for pre-existing simulations.
            Any already-simulated pointings will be skipped.
        """
        # read the observations from the sql database
        with sqlite3.connect(opsimPath) as conn:
            observations = pd.read_sql(
                f"select {', '.join(self._columns.keys())} from observations;", conn
            )

        # rename the columns
        observations = observations.rename(columns=self._columns)

        # convert to an Astropy table with units
        observations = table.Table.from_pandas(observations, units=self._units)

        # save the observations
        self.observations = observations

        # keep a list of the remaining indices
        self._remaining = list(np.arange(len(observations)))

        # if checkDir provided, remove previous pointings from the remaining list
        if checkDir is not None:
            files = glob(f"{checkDir}/dof/*")
            old_pointings = [
                int(file.split("/")[-1].split(".")[0][3:]) for file in files
            ]
            self._remaining = [i for i in self._remaining if i not in old_pointings]

    def getRandomObservation(self, rng: np.random.Generator) -> table.Row:
        """Get a random observation from the database of observations.

        Parameters
        ----------
        rng: np.random.Generator
            A numpy random generator.

        Returns
        -------
        astropy.table.Row
            The random observation in an Astropy table.
        """
        # if we've already used all the observations, raise an error
        if len(self._remaining) == 0:
            raise RuntimeError("No more unique observations left.")

        # randomly select one of the remaining observations
        remaining_idx = rng.integers(len(self._remaining))
        observation_idx = self._remaining.pop(remaining_idx)
        return self.observations[observation_idx]
