# BlurCropperDetector
Detect and crop out image blur (at least, images of sediment and coins); a trial for the SandSnap project

0. Create a conda environment

A. Conda housekeeping

`conda clean --all`
`conda update -n base -c defaults conda`

B. Create new `blur` conda environment

We'll create a new conda environment and install packages into it from conda-forge

`conda env create -f install/blur.yml`

activate:

`conda activate blur`

1. Run the jupyter notebook kernel from the command line

`ipython notebook`

2. Open and run the notebook
