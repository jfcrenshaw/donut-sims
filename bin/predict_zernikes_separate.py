# set the mode to either "ml" or "ts_wep"
mode = "ml"

# general imports
import numpy as np

# imports to load the simulated data
from ml_aos.dataloader import JFsDonuts as Donuts
from torch.utils.data import DataLoader
from astropy.table import Table
import galsim

# load the libraries for the ML method
if mode == "ml":
    # imports to load the trained ML model
    from ml_aos.david_net import DavidNet
    import torch

    # load the trained model
    # ckpt_path = "/phys/users/jfc20/ml-aos/experiments/resilient-breeze-27/lightning_logs/er3y3ag6/checkpoints/epoch=49-step=10950.ckpt"
    ckpt_path = "/phys/users/jfc20/ml-aos/fiery-forest-30/lightning_logs/3gqd7kka/checkpoints/epoch=18-step=8892.ckpt"
    model = DavidNet(n_meta_layers=3, input_shape=170)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model.eval()

# load the libraries for the ts_wep method
elif mode == "ts_wep":
    # imports for ts_wep
    from lsst.ts.wep.cwfs.Algorithm import Algorithm
    from lsst.ts.wep.cwfs.CompensableImage import CompensableImage
    from lsst.ts.wep.cwfs.Instrument import Instrument
    from lsst.ts.wep.Utility import CamType, DefocalType, getConfigDir
    import os

    # ts_wep configs
    cwfsConfigDir = os.path.join(getConfigDir(), "cwfs")
    instDir = os.path.join(cwfsConfigDir, "instData")
    inst = Instrument()
    instConfigFile = os.path.join(instDir, "lsst", "instParamPipeConfig.yaml")
    maskConfigFile = os.path.join(instDir, "lsst", "maskMigrate.yaml")
    inst.configFromFile(170, CamType.LsstCam, instConfigFile, maskConfigFile)
    algoDir = os.path.join(cwfsConfigDir, "algo")


# conversion factors for the Zernikes
arcsec_per_micron = np.array(
    [
        0.751,  # Z4
        0.271,  # Z5
        0.271,  # Z6
        0.819,  # Z7
        0.819,  # Z8
        0.396,  # Z9
        0.396,  # Z10
        1.679,  # Z11
        0.937,  # Z12
        0.937,  # Z13
        0.517,  # Z14
        0.517,  # Z15
        1.755,  # Z16
        1.755,  # Z17
        1.089,  # Z18
        1.089,  # Z19
        0.635,  # Z20
        0.635,  # Z21
        2.810,  # Z22
    ]
)

# create lists to store everything in
pntId = []
obsId = []
intraObjId = []
extraObjId = []
z_true = []
z_tswep = []
z_ml_intra = []
z_ml_extra = []

# loop over the data set
data_dir = "/astro/store/epyc/users/jfc20/aos_sims"
donutSet = Donuts("test", data_dir=data_dir)
for donuts in DataLoader(donutSet, batch_size=64, drop_last=True):
    # get list of unique pointings
    pointings = list(set(donuts["pntId"].numpy()))

    # get pointing flags
    pointing_flags = [(donuts["pntId"] == pnt).numpy().flatten() for pnt in pointings]

    # get sensor flags
    sensors = ["R00", "R40", "R04", "R44"]
    sensor_flags = [
        ((donuts["field_x"] < 0) & (donuts["field_y"] < 0)).numpy().flatten(),
        ((donuts["field_x"] < 0) & (donuts["field_y"] > 0)).numpy().flatten(),
        ((donuts["field_x"] > 0) & (donuts["field_y"] < 0)).numpy().flatten(),
        ((donuts["field_x"] > 0) & (donuts["field_y"] > 0)).numpy().flatten(),
    ]

    # get the intra/extrafocal flags
    focal_flag = donuts["intrafocal"].numpy().flatten().astype(bool)

    if mode == "ml":
        # let's go ahead and predict zernikes for the whole batch using the ML algorithm
        with torch.no_grad():
            zml = model(
                donuts["image"],
                donuts["field_x"],
                donuts["field_y"],
                donuts["intrafocal"],
            ).numpy()

    # loop over pointings
    for pflag in pointing_flags:
        # loop over sensors
        for sflag in sensor_flags:
            # get indices of the intra and extrafocal donuts
            intra_idx = np.where(pflag & sflag & focal_flag)[0]
            extra_idx = np.where(pflag & sflag & ~focal_flag)[0]

            # loop over pairs
            for intra, extra in zip(intra_idx, extra_idx):
                # get the true zernikes
                # they're identical, so it shouldn't matter which one we use
                assert np.allclose(
                    donuts["zernikes"][intra].numpy(),
                    donuts["zernikes"][extra].numpy(),
                )
                zt = donuts["zernikes"][intra].numpy()

                if mode == "ts_wep":
                    try:
                        # get the LSST band
                        opSimTable = Table.read(f"{data_dir}/opSimTable.parquet")
                        obsRow = opSimTable[
                            opSimTable["observationId"]
                            == donuts["obsId"].numpy()[intra]
                        ][0]
                        band = obsRow["lsstFilter"]

                        # predict zernikes using ts_wep...
                        # ---------------------------------------------------
                        intra_img = CompensableImage()
                        intra_img.setImg(
                            np.rad2deg(
                                [
                                    donuts["field_x"][intra].numpy()[0],
                                    donuts["field_y"][intra].numpy()[0],
                                ]
                            ),
                            DefocalType.Intra,
                            image=donuts["image"][intra][0].numpy().copy(),
                        )

                        extra_img = CompensableImage()
                        extra_img.setImg(
                            np.rad2deg(
                                [
                                    donuts["field_x"][extra].numpy()[0],
                                    donuts["field_y"][extra].numpy()[0],
                                ]
                            ),
                            DefocalType.Extra,
                            image=donuts["image"][extra][0].numpy().copy(),
                        )

                        expAlgo = Algorithm(algoDir)
                        expAlgo.config("exp", inst)
                        expAlgo.runIt(
                            I1=intra_img, I2=extra_img, model="offAxis", tol=1e-4
                        )
                        zp = expAlgo.getZer4UpInNm()

                        # convert nm --> microns
                        zp /= 1e3

                        # weight by the effective wavelength in microns
                        bandpass = galsim.Bandpass(f"LSST_{band}.dat", wave_type="nm")
                        zp *= bandpass.effective_wavelength / 1e3

                        # convert microns --> PSF contribution
                        zp *= arcsec_per_micron
                        # ---------------------------------------------------
                    except:
                        zp = np.full_like(arcsec_per_micron, np.nan)

                # save everything in the lists
                pntId.append(donuts["pntId"].numpy()[intra])
                obsId.append(donuts["obsId"].numpy()[intra])
                intraObjId.append(donuts["objId"].numpy()[intra])
                extraObjId.append(donuts["objId"].numpy()[extra])
                z_true.append(zt)
                if mode == "ml":
                    z_ml_intra.append(zml[intra])
                    z_ml_extra.append(zml[extra])
                elif mode == "ts_wep":
                    z_tswep.append(zp)

# convert the lists to numpy arrays
pntId = np.array(pntId)  # type: ignore
obsId = np.array(obsId)  # type: ignore
intraObjId = np.array(intraObjId)  # type: ignore
extraObjId = np.array(extraObjId)  # type: ignore
z_true = np.array(z_true)  # type: ignore
z_tswep = np.array(z_tswep)  # type: ignore
z_ml_intra = np.array(z_ml_intra)  # type: ignore
z_ml_extra = np.array(z_ml_extra)  # type: ignore

# save everything!
np.savez(
    f"../data/zernike_predictions_{mode}.npz",
    pntId=pntId,
    obsId=obsId,
    intraObjId=intraObjId,
    extraObjId=extraObjId,
    z_true=z_true,
    z_tswep=z_tswep,
    z_ml_intra=z_ml_intra,
    z_ml_extra=z_ml_extra,
)
