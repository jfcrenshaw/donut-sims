"""Simulate donuts but don't save in the butler."""
import time
from pathlib import Path

import numpy as np

from donut_sims import ObsSimulator, dofsToZernikes, generateDOF

# set the directories where we will save the results
SAVE_DIR = Path("/astro/store/epyc/users/jfc20/aos_sims")
CATALOG_DIR = SAVE_DIR / "catalogs"
ZERNIKE_DIR = SAVE_DIR / "zernikes"
DOF_DIR = SAVE_DIR / "dofs"
IMAGE_DIR = SAVE_DIR / "images"

# create these directories if they do not exist
CATALOG_DIR.mkdir(parents=True)
ZERNIKE_DIR.mkdir(parents=True)
DOF_DIR.mkdir(parents=True)
IMAGE_DIR.mkdir(parents=True)

# start timing the simulations
sims_start = time.time()

for seed in range(1000):

    # start timing this simulation
    start = time.time()

    name = f"seed={seed}"
    print(name)

    # set the rng
    rng = np.random.default_rng(seed)

    # random degrees of freedom to perturb the telescope
    print("Generating DOFs...")
    dof = generateDOF(rng, norm=0.25)

    # simulate the donuts
    print("Simulating observation...")
    obsSimulator = ObsSimulator()
    observation = obsSimulator.simulateObs(dof, rng)

    # save the simulation
    obsId = observation["metadata"]["observationId"]

    # save the catalog
    observation["catalog"].write(CATALOG_DIR / f"obs{obsId}.catalog.parquet")

    # save the zernikes
    for detector in ["R00", "R40", "R44", "R04"]:
        zernikes = dofsToZernikes(
            dof, f"{detector}_SW0", observation["metadata"]["lsstFilter"]
        )
        np.save(
            ZERNIKE_DIR / f"obs{obsId}.detector{detector[:3]}.zernikes.npy", zernikes
        )

    # save the dofs
    np.save(DOF_DIR / f"obs{obsId}.dofs.npy", dof)

    # save the images
    for objId, img in observation["images"].items():
        np.save(IMAGE_DIR / f"obs{obsId}.obj{objId}.image.npy", img.T)

    # we're done with this one!
    print("Done with", obsId)

    # print how long this simulation took
    elapsed = time.time() - start
    print("Elapsed time", time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed)), "\n")

# save the observation database
obsSimulator.obsScheduler.observations.write(SAVE_DIR / "observations.parquet")

# figure out how long the whole thing took
sims_end = time.time()
elapsed = sims_end - sims_start
print("script time", time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed)))
