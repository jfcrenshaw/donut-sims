# donut-sims

Donut simulations for the Rubin AOS Wavefront estimation pipeline.

To install:

1. source the LSST stack, including `ts_wep` and `phosim_utils`:

```bash
source lsst_setup.sh
```

Note that you might need to change the paths listed in that file to match paths on your machine.

2. Install the donut-sim package:

```bash
pip install -e .
```

When installing, if `batoid` complains that CMake is too old, you probably need to add a symlink to `cmake3` in your local bin.
For example, on my system

```shell
ln -s /usr/bin/cmake3 ~/.local/bin/cmake
```

NOTE:
The pip install seems to work, and a donut-sims package appears in my local site packages, but I still cannot import `donut_sims`.
You can get around this by running scripts in the root directory of donut-sims, but I will try to fix this in the future.
