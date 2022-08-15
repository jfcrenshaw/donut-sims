"""Class to select observations from OpSim."""
import sqlite3

import astropy.units as u
import numpy as np
import pandas as pd
from astropy import table


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
        opsim_path: str = "/astro/store/epyc/users/jfc20/rubin_sim_data/sim_baseline/baseline.db",
    ) -> None:
        """
        Parameters
        ----------
        opsim_path: str
            Path to the OpSim simulation database.
        """
        # read the observations from the sql database
        with sqlite3.connect(opsim_path) as conn:
            observations = pd.read_sql(
                f"select {', '.join(self._columns.keys())} from observations;", conn
            )

        # rename the columns
        observations = observations.rename(columns=self._columns)

        # convert to an Astropy table with units
        observations = table.Table.from_pandas(observations, units=self._units)

        # save the observations
        self.observations = observations

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
        idx = rng.integers(len(self.observations))
        return self.observations[idx]
