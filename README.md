# Code for "The ozone radiative forcing of nitrogen oxide emissions from aviation can be estimated using a probabilistic approach" by Rao et. al. (2024).

The research entails two critical aspects: Firstly, the training data from chemistry-climate situations that was processed using Matlab code. And secondly, the Gaussian process regression (GPR) models to reproduce these predictions using Python with the help of notebooks and scripts. The predictions from GPR models are entirely reproducible with the given Conda environment, which is the essence of the work. Other aspects such as the convergence study are not currently included in the repository.

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/pratikvrao/GPR_code.git
cd GPR_code
conda env create -f environment.yml
conda activate myenv


## Structure

GPR_code/
│
├── core/
│   ├── estimate_pdf.py
│   ├── GPR_build.py
│   ├── preprocess.py
│   └── load_trajectory_data.py
│
├── data_for_training/
│   ├── iRF_global_trg.mat
│   └── weather_global_trg.mat
│
├── trajectory_data/
│   ├── traj1.mat
│   ├── traj2.mat
│   ├── traj3.mat
│   └── traj4_AirTraf.mat
│
├── GPR_nature_cleaned_version.ipynb
│
└── README.md

## Notes

There is one python notebook used to train and test both the standard and chained GPR models. The former runs very quickly, but the latter takes about 20 minutes. The notebook uses various python scripts from the core directory and can be used later to predict the ozone RF of arbitrary flight paths for cruise level. The example data from the paper are included in the repository. The models can be retrained for new data.
 
#TODO

- Saving a copy of both GPR models so that they can be used directly for testing or predicting the ozone RF of arbitrary flights.
- Providing Matlab code for EUROCONTROL data
