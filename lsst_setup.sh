
# load the LSST conda environment
source /astro/store/epyc/projects/lsst_comm/lsstinstall/loadLSST.bash

# activate the lsst_distrib top-level package
setup lsst_distrib

# setup the AOS packages
setup -k -r /astro/store/epyc/users/jfc20/lsst/ts_wep
setup -k -r /astro/store/epyc/users/jfc20/lsst/phosim_utils

# export path to rubin_sim data
export RUBIN_SIM_DATA_DIR=/astro/store/epyc/users/jfc20/rubin_sim_data