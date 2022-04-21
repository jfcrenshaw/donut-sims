# donut-sims

Donut simulations for the Rubin AOS Wavefront estimation pipeline.

To install:

1. Run `conda env create -f environment.yml`
2. Activate the new environment via `conda activate donut-sims`
3. Run `poetry install`

When installing, if `batoid` complains that CMake is too old, you probably need to add a symlink to `cmake3` in your local bin.
For example, on my system

```shell
ln -s /usr/bin/cmake3 ~/.local/bin/cmake
```
