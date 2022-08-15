"""Test script to simulate a small set of simulations.

Using this to quickly test that everything is working.
"""
import time

import numpy as np

from donut_sims import ObsSimulator, SimsToButler, generateDOF

script_start = time.time()

for seed in range(200, 300):
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

    # save the donuts in the butler
    print("Saving sims in butler...")
    stb = SimsToButler()
    stb.saveSimulation(name, observation, dof)

    elapsed = time.time() - start
    print("Elapsed time", time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed)), "\n")

script_end = time.time()
elapsed = script_end - script_start
print("script time", time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed)))
