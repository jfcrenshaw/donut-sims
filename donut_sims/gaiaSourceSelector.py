"""Generate a catalog of Gaia sources given a pointing."""
from pathlib import Path
from typing import List

import astropy.units as u
import galsim
import lsst.daf.butler as dafButler
import lsst.geom
import numpy as np
import wfsim
from astropy import table
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Camera, Detector
from lsst.meas.algorithms import LoadReferenceObjectsConfig, ReferenceObjectLoader
from lsst.ts.wep.ParamReader import ParamReader
from lsst.ts.wep.task.RefCatalogInterface import RefCatalogInterface
from lsst.ts.wep.Utility import getConfigDir
from sklearn.neighbors import NearestNeighbors


class GaiaSourceSelector:
    """Select Gaia sources for Rubin imaging."""

    gaiaColumns = {
        "id": "objectId",
        "coord_ra": "ra",
        "coord_dec": "dec",
        "centroid_x": "xCentroid",
        "centroid_y": "yCentroid",
        "phot_g_mean_flux": "gaiaG",
        "phot_bp_mean_flux": "gaiaBp",
        "phot_rp_mean_flux": "gaiaRp",
    }

    def __init__(
        self,
        butlerRepo: str = "/epyc/data/lsst_refcats/gen3",
        collections: List[str] = ["refcats"],
        refCatName: str = "gaia_dr2_20200414",
        donutRadius: float = 75,
    ) -> None:
        """
        Parameters
        ----------
        butlerRepo: str, default="/epyc/data/lsst_refcats/gen3"
            Path to the butler repository.
        collections: list, default=["refcats"]
            The collections to be searched when loading the data.
        refCatName: str, default="gaia_dr2_20200414"
            Name of the reference catalog.
        donutRadius: float, default=75
            The donut radius used to determine whether donuts are blended.
        """
        # save the butler and collections
        self._butler = dafButler.Butler(butlerRepo, collections=collections)

        # save the camera
        self._camera = self.butler.get(
            "camera",
            dataId={"instrument": "LSSTCam"},
            collections=["LSSTCam/calib/unbounded"],
        )

        # save the reference catalog name
        self.refCatName = refCatName

        # and the donut radius used for AOS source selection
        self.donutRadius = donutRadius

        # load the Lsst r bandpass to use for magnitude conversions
        self._lsstRBand = galsim.Bandpass("LSST_r.dat", wave_type="nm").withZeropoint(
            "AB"
        )

    @property
    def butler(self) -> dafButler.Butler:
        """Return the Butler."""
        return self._butler

    @property
    def camera(self) -> Camera:
        """Return the LSST Camera."""
        return self._camera

    def _getRefObjLoader(
        self, refCatInterface: RefCatalogInterface
    ) -> ReferenceObjectLoader:
        """Return the reference object loader."""
        # get IDs of hierarchical triangular mesh (HTM) shards
        # covering the field of view
        htmIds = refCatInterface.getHtmIds()

        # get the data references and IDs for this pointing
        dataRefs, dataIds = refCatInterface.getDataRefs(
            htmIds,
            self.butler,
            self.refCatName,
            self.butler.collections,
        )

        refObjLoader = ReferenceObjectLoader(
            dataIds=dataIds,
            refCats=dataRefs,
            config=LoadReferenceObjectsConfig(),
        )

        return refObjLoader

    def _removeBadFluxes(self, catalog: table.Table) -> table.Table:
        """Remove negative fluxes and NaNs from the catalog.

        nan_to_num maps NaNs to zero, so these are removed by the positivity cut.

        Parameters
        ----------
        catalog: astropy.table.Table
            Astropy table of the catalog.

        Returns
        -------
        astropy.table.Table
            The catalog with bad fluxes removed.
        """
        idx = np.where(
            (np.nan_to_num(catalog["gaiaG"]) > 0.0)
            & (np.nan_to_num(catalog["gaiaBp"]) > 0.0)
            & (np.nan_to_num(catalog["gaiaRp"]) > 0.0)
        )

        return catalog[idx]

    def _convertFluxesToLsstMag(
        self,
        catalog: table.Table,
        lsstFilter: str,
    ) -> table.Table:
        """Convert the catalog fluxes to the requested LSST filter.

        This works by converting Gaia magnitudes to an SDSS r band magnitude
        via the formula given in Table 5.7 here:
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/
            chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html

        We assume SDSS r ~ LSST r, then use the SED for each star to translate
        the LSST r magnitude to other magnitudes.

        Parameters
        ----------
        catalog: astropy.table.Table
            Astropy table of the catalog.
        lsstFilter: str
            The name of the LSST filter for which the magnitude is calculated.
            Saved as `lsstMag`.

        Returns
        -------
        astropy.table.Table
            The new catalog with the flux columns replaced by `lsstMag`.
        """
        # first calculate SDSS r
        xGaia = (catalog["gaiaBp"].to(u.ABmag) - catalog["gaiaRp"].to(u.ABmag)).value
        gGaia = catalog["gaiaG"].to(u.ABmag).value
        rSdss = (
            +gGaia
            + 0.12879
            + -0.24662 * xGaia
            + 0.027464 * xGaia**2
            + 0.049465 * xGaia**3
        ) * u.ABmag

        # we will assume rLsst ~ rSdss
        rLsst = rSdss

        # convert to requested band
        if lsstFilter == "r":
            lsstMag = rLsst
        else:
            # list to hold the new magnitudes
            lsstMag = []
            # bandpass we are converting to
            bandpass = galsim.Bandpass(
                f"LSST_{lsstFilter}.dat", wave_type="nm"
            ).withZeropoint("AB")

            # use SEDs to calculate the magnitude in the requested band
            for r, T in zip(rLsst, catalog["temperature"]):
                sed = wfsim.BBSED(T).withMagnitude(r.value, self._lsstRBand)
                lsstMag.append(sed.calculateMagnitude(bandpass))

            # add the astropy unit for AB mags
            lsstMag *= u.ABmag

        # add the LSST magnitude to the catalog
        catalog["lsstMag"] = lsstMag

        # and remove the gaia magnitudes
        catalog.remove_columns(["gaiaG", "gaiaBp", "gaiaRp"])

        return catalog

    def _selectAosSources(
        self,
        catalog: table.Table,
        lsstFilter: str,
        detector: Detector,
        maxSources: int = 10,
    ) -> table.Table:
        """Select the AOS sources and remove donuts that do not blend with sources.

        Parameters
        ----------
        catalog: astropy.table.Table
            An astropy table representing the catalog.
        lsstFilter: str
            The LSST filter that magnitude cuts are performed in.
        detector: lsst.afw.cameraGeom.Detector
            The detector on which this catalog is imaged.
        maxSources: int, default=10
            The maximum number of AOS sources.

        Returns
        -------
            The catalog with AOS sources and blends flagged, and with extraneous
            donuts removed.
        """

        # load the AOS magnitude range for each band
        magLimPath = Path(getConfigDir()) / "task" / "magLimitStar.yaml"
        magLimFile = ParamReader(filePath=magLimPath)
        magLim = magLimFile.getSetting(f"filter{lsstFilter.upper()}")

        # make the source magnitude cut
        magCut = (catalog["lsstMag"] >= magLim["low"]) & (
            catalog["lsstMag"] <= magLim["high"]
        )

        # select donuts that don't fall off the detector
        boundaryCut = (
            detector.getBBox()
            .erodedBy(int(2 * self.donutRadius))
            .contains(catalog["xCentroid"], catalog["yCentroid"])
        )

        # select donuts that are either isolated or the brightest in their blend
        xy = np.vstack((catalog["xCentroid"], catalog["yCentroid"])).T
        xyNeigh = NearestNeighbors(radius=2 * self.donutRadius)
        xyNeigh.fit(xy)
        radDist, radIdx = xyNeigh.radius_neighbors(xy, sort_results=True)
        blendCut = []
        for neighbors in radIdx:
            mags = catalog["lsstMag"][neighbors]
            blendCut.append(all(mags[0] < mags[1:]))

        # sources that pass all of these cuts will be used for the AOS
        catalog["aosSource"] = magCut & boundaryCut & blendCut

        # if the total number of aosSources is greater than maxSources
        # select the brightest sources
        if (nSources := np.sum(catalog["aosSource"])) > maxSources:
            # make a mask for all of the AOS sources
            mask = np.array(nSources * [False])

            # select the N brightest sources (N = maxSources)
            brightest = np.argpartition(
                catalog[catalog["aosSource"]]["lsstMag"], -maxSources
            )[-maxSources:]
            mask[brightest] = True

            # mask out all aosSources except for the brightest ones
            catalog["aosSource"][catalog["aosSource"]] = mask

        # label donuts that are blends with AOS sources
        blendId = -np.ones(len(catalog), dtype=int)
        if any(catalog["aosSource"]):
            # get overlaps with AOS sources
            _, radIdx = xyNeigh.radius_neighbors(
                xy[catalog["aosSource"]], sort_results=True
            )

            # create blendIds
            for neighbors in radIdx:
                if len(neighbors) > 1:
                    blendId[neighbors] = catalog["objectId"][neighbors[0]]

        catalog["blendId"] = blendId

        # return only sources that are AOS sources or that blend with AOS sources
        return catalog[catalog["aosSource"] | (catalog["blendId"] > 0)]

    def selectSources(
        self,
        boresightRA: float,
        boresightDec: float,
        boresightRotAng: float,
        lsstFilter: str = "r",
        detectorNames: List[str] = None,
        selectAosSources: bool = True,
        maxAosSources: int = 10,
        rng: np.random.Generator = None,
    ) -> table.Table:
        """Select source catalog corresponding to the given pointing.

        Parameters
        ----------
        boresightRA: float
            Right ascension (RA) of the telescope pointing in degrees.
        boresightDec: float
            Declination (Dec) of the telescope pointing in degrees.
        boresightRotAng: float
            Rotation angle of the telescope about the optical axis in degrees.
        lsstFilter: str, default="r"
            The LSST filter in which magnitudes are calculated.
        detectorNames: list, optional
            The list of detectors for which to generate the source catalog.
            If None, defaults to all corner wavefront sensors.
        selectAosSources: bool, default=True
            Whether to select the AOS sources and remove donuts that do not
            blend with these sources.
        maxAosSources: int, default=10
            The maximum number of AOS sources per detector.
        rng: np.random.Generator, optional
            A numpy random number generator, used to simulate random temperatures
            for stars. If not provided, np.random.default_rng(0) is used.

        Returns
        -------
        astropy.table.Table
            An astropy table of the sources.
        """
        # if no rng is passed, use zero for the random seed
        rng = np.random.default_rng(0) if rng is None else rng

        # load the reference catalog interface for this pointing
        refCatInterface = RefCatalogInterface(
            boresightRA, boresightDec, boresightRotAng
        )

        # get the reference object loader
        refObjLoader = self._getRefObjLoader(refCatInterface)

        # if detectorName is None, get the default list
        if detectorNames is None:
            detectorNames = [
                f"R{i}_SW{j}" for i in ["00", "40", "44", "04"] for j in [0, 1]
            ]

        # loop over the detectors and generate catalogs
        catalog = []
        for name in detectorNames:
            # load the detector
            detector = self.camera[name]

            # Get the refcatalog shard
            skyBox = refObjLoader.loadPixelBox(
                detector.getBBox(),
                refCatInterface.getDetectorWcs(detector),
                filterName="phot_g_mean",
                bboxToSpherePadding=0,
            )

            # get the catalog of stars
            cat = skyBox.refCat.asAstropy()

            # get the requested columns and rename them
            cat = cat[list(self.gaiaColumns.keys())]
            cat.rename_columns(
                list(self.gaiaColumns.keys()),
                list(self.gaiaColumns.values()),
            )

            # remove bad fluxes
            cat = self._removeBadFluxes(cat)

            # random temperatures
            temperatures = rng.uniform(4_000, 10_000, size=len(cat))
            cat["temperature"] = temperatures
            cat["temperature"].unit = u.K

            # convert fluxes to LSST magnitude
            cat = self._convertFluxesToLsstMag(cat, lsstFilter)

            # select AOS sources
            if selectAosSources:
                cat = self._selectAosSources(cat, lsstFilter, detector, maxAosSources)

            # if the catalog is empty, move on
            if len(cat) == 0:
                continue

            # save the detector name
            cat["detector"] = len(cat) * [name]

            # convert pixel numbers to field angles
            pixelPoints = [
                lsst.geom.Point2D(cenx, ceny)
                for cenx, ceny in zip(cat["xCentroid"], cat["yCentroid"])
            ]
            fieldAngles = np.array(
                self.camera.transform(
                    pixelPoints, detector.makeCameraSys(PIXELS), FIELD_ANGLE
                )
            )
            cat["xField"] = fieldAngles[:, 0]
            cat["yField"] = fieldAngles[:, 1]

            catalog.append(cat)

        # combine all catalogs into one
        catalog = table.vstack(catalog)

        # reorder the columns
        catalog = catalog[  # type: ignore
            [
                "objectId",
                "blendId",
                "aosSource",
                "ra",
                "dec",
                "xField",
                "yField",
                "xCentroid",
                "yCentroid",
                "detector",
                "lsstMag",
                "temperature",
            ]
        ]

        # clip the magnitudes for efficiency
        # I will remove this later
        catalog["lsstMag"] = np.clip(catalog["lsstMag"], 14, None)

        return catalog
