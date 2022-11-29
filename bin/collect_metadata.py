"""This script loops through all of the objects for which we have zernike 
predictions and collects the metadata we want to plot vs prediction error."""
from astropy.table import Table
import numpy as np
from sklearn.mixture import GaussianMixture

# load the ML predictions
ml_data = np.load("../data/zernike_predictions_ml.npz")

# create dictionary to save the metadata
meta = {
    "pntId": [],
    "obsId": [],
    "intraObjId": [],
    "extraObjId": [],
    "lsst_filter": [],
    "airmass": [],
    "seeing": [],
    "sensor": [],
    "field_angle": [],
    "intra_snr": [],
    "intra_faint_blends": [],
    "intra_bright_blends": [],
    "extra_snr": [],
    "extra_faint_blends": [],
    "extra_bright_blends": [],
}

# load the Rubin OpSim table
opSimTable = Table.read("/astro/store/epyc/users/jfc20/aos_sims/opSimTable.parquet")

# loop through all of the objects from the ML predictions
for pnt, obs, intraObj, extraObj in zip(
    ml_data["pntId"],
    ml_data["obsId"],
    ml_data["intraObjId"],
    ml_data["extraObjId"],
):
    # append the IDs
    meta["pntId"].append(pnt)
    meta["obsId"].append(obs)
    meta["intraObjId"].append(intraObj)
    meta["extraObjId"].append(extraObj)

    # append the observing metadata
    obsRow = opSimTable[opSimTable["observationId"] == obs][0]
    meta["lsst_filter"].append(obsRow["lsstFilter"])
    meta["airmass"].append(obsRow["airmass"])
    meta["seeing"].append(obsRow["seeingFwhm500"])

    # open the catalog for this object
    cat = Table.read(
        f"/astro/store/epyc/users/jfc20/aos_sims/catalogs/pnt{pnt}.catalog.parquet"
    )

    # save the sensor name
    intra = cat[cat["objectId"] == intraObj][0]
    meta["sensor"].append(intra["detector"][:3])

    # calculate the field angle
    fx, fy = intra["xField"], intra["yField"]
    angle = np.sqrt(fx**2 + fy**2)
    meta["field_angle"].append(np.rad2deg(angle))

    # count the number of intrafocal blending neighbors
    obj = intraObj
    central = cat[cat["objectId"] == obj][0]
    blends = cat[(cat["blendId"] == obj) & (cat["aosSource"] == False)]
    faint_blends = blends[blends["lsstMag"] > central["lsstMag"] + 2]
    bright_blends = blends[blends["lsstMag"] <= central["lsstMag"] + 2]
    meta["intra_faint_blends"].append(len(faint_blends))
    meta["intra_bright_blends"].append(len(bright_blends))

    # calculate the intrafocal SNR
    if len(faint_blends) + len(bright_blends) > 0:
        # if the star is blended, snr is undefined
        snr = np.nan
    else:
        # fit a 2-component gaussian mixture model to the pixel brightnesses
        img = np.load(
            f"/astro/store/epyc/users/jfc20/aos_sims/images/pnt{pnt}.obs{obs}.obj{obj}.image.npy"
        )
        gm = GaussianMixture(2, random_state=0).fit(img.flatten().reshape(-1, 1))

        # get the mean of the brighter gaussian, and the std of the dimmer gaussian
        idx = gm.means_.flatten().argsort()
        signal = gm.means_[idx[-1]][0]
        noise = np.sqrt(gm.covariances_[idx[0]][0][0])

        # calculate snr
        snr = signal / noise

    meta["intra_snr"].append(snr)

    # count the number of extrafocal blending neighbors
    obj = extraObj
    central = cat[cat["objectId"] == obj][0]
    blends = cat[(cat["blendId"] == obj) & (cat["aosSource"] == False)]
    faint_blends = blends[blends["lsstMag"] > central["lsstMag"] + 2]
    bright_blends = blends[blends["lsstMag"] <= central["lsstMag"] + 2]
    meta["extra_faint_blends"].append(len(faint_blends))
    meta["extra_bright_blends"].append(len(bright_blends))

    # calculate the extrafocal SNR
    if len(faint_blends) + len(bright_blends) > 0:
        # if the star is blended, snr is undefined
        snr = np.nan
    else:
        # fit a 2-component gaussian mixture model to the pixel brightnesses
        img = np.load(
            f"/astro/store/epyc/users/jfc20/aos_sims/images/pnt{pnt}.obs{obs}.obj{obj}.image.npy"
        )
        gm = GaussianMixture(2, random_state=0).fit(img.flatten().reshape(-1, 1))

        # get the mean of the brighter gaussian, and the std of the dimmer gaussian
        idx = gm.means_.flatten().argsort()
        signal = gm.means_[idx[-1]][0]
        noise = np.sqrt(gm.covariances_[idx[0]][0][0])

        # calculate snr
        snr = signal / noise

    meta["extra_snr"].append(snr)

# convert everything to numpy arrays
meta = {key: np.array(value) for key, value in meta.items()}

# save everything in a numpy file
np.savez("../data/metadata.npz", **meta)
