# Code supporting the thesis
This directory contains code producing all `.pdf` and `.pgf` figures found in `../[pdf,pgf]` as well as some tabular material formatted as LaTeX source code in `../../data/`.
It can also run all simulations supporting the the thesis. To do this, set the `RUN_SIMULATION` environment variable to `"True"`. To freshly extract data from the `QCoDeS` database files (requires access to the RWTH network and the AG Bluhm server), set the `EXTRACT_DATA` environment variable. All required data is already extracted to `.hdf5` format and stored in `../../data`, however.

## Running the code
The code is pure Python, except for an optional dependency on MATLAB, specifically the `logm_frechet` [package](https://www.mathworks.com/matlabcentral/fileexchange/38894-matrix-logarithm-with-frechet-derivatives-and-condition-number) and [Matrix Function toolbox](https://www.mathworks.com/matlabcentral/fileexchange/20820-the-matrix-function-toolbox), to compute confidence intervals in `filter_functions/monte_carlo_filter_functions.py`.
All dependencies are open-source except for `poisson_schroedinger_1d`, which requires access to the `qutech` group at `https://git-ce.rwth-aachen.de/qutech/`.

Dependencies are managed using [`pixi`](https://pixi.sh/), which is the only executable you will need to run the code.
The final version of the thesis contains figures and data produced by dependency versions stated in `pixi.lock`.

You can either run scripts individually,
```sh
pixi run python setup/imaging.py
```
or run multiple (or all) scripts in one go,
```sh
pixi run python run.py --all --parallel
```
Run 
```sh
pixi run python run.py --help
``` 
for an overview of options.

To include the `logm_frechet` derivative in `filter_functions/monte_carlo_filter_functions.py`, specify the `logm-frechet` feature when calling `pixi`:
```sh
pixi run --environment logm-frechet python run.py 
```