"""Simulate donuts but don't save in the butler."""
import time
from pathlib import Path
import numpy as np
from donut_sims import ObsSimulator
from donut_sims import dofsToZernikes
from donut_sims import generateDOF
from datetime import timedelta

# set the directories where we will save the results
SAVE_DIR = Path("/astro/store/epyc/users/jfc20/aos_sims")
CATALOG_DIR = SAVE_DIR / "catalogs"
ZERNIKE_DIR = SAVE_DIR / "zernikes"
DOF_DIR = SAVE_DIR / "dof"
IMAGE_DIR = SAVE_DIR / "images"

# create these directories
CATALOG_DIR.mkdir(parents=True, exist_ok=True)
ZERNIKE_DIR.mkdir(parents=True, exist_ok=True)
DOF_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# start timing the simulations
sims_start = time.time()

print("Creating the ObsSimulator...\n")
obsSimulator = ObsSimulator(checkDir=str(SAVE_DIR))

# save the observation database
obsSimulator.obsScheduler.observations.write(SAVE_DIR / "opSimTable.parquet")


for seed in range(10_000):

    # start timing this simulation
    start = time.time()

    name = f"seed={seed}"
    print(name)

    # set the rng
    rng = np.random.default_rng(seed)

    # random degrees of freedom to perturb the telescope
    print("Generating DOFs...")
    dof = generateDOF(rng, norm=0.45)

    # simulate the donuts
    print("Simulating observation...")
    recomputeAtm = seed % 100 == 0
    observation = obsSimulator.simulateObs(dof, rng, recomputeAtm=recomputeAtm)

    # get the metadata
    pntId = observation["catalog"]["pointingId"][0]  # type: ignore
    obsId = observation["catalog"]["observationId"][0]  # type: ignore

    # save the catalog
    observation["catalog"].write(CATALOG_DIR / f"pnt{pntId}.catalog.parquet")  # type: ignore

    # save the zernikes
    obs_table = obsSimulator.obsScheduler.observations
    lsstFilter = obs_table[obs_table["observationId"] == obsId]["lsstFilter"][0]
    for detector in ["R00", "R40", "R44", "R04"]:
        zernikes = dofsToZernikes(dof, f"{detector}_SW0", lsstFilter)
        np.save(
            ZERNIKE_DIR / f"pnt{pntId}.obs{obsId}.detector{detector[:3]}.zernikes.npy",
            zernikes,
        )

    # save the dofs
    np.save(DOF_DIR / f"pnt{pntId}.dofs.npy", dof)

    # save the images
    for objId, img in observation["images"].items():  # type: ignore
        np.save(IMAGE_DIR / f"pnt{pntId}.obs{obsId}.obj{objId}.image.npy", img)

    # we're done with this one!
    print(f"Done with pnt{pntId} obs{obsId}")

    # print how long this simulation took
    elapsed = time.time() - start
    print("Elapsed time", time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed)), "\n")

# figure out how long the whole thing took
sims_end = time.time()
elapsed = sims_end - sims_start
print("script time", timedelta(seconds=elapsed))
