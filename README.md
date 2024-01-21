# Interpretable Machine Learning Enabled Inorganic Reaction Classification and Synthesis Condition Prediction
This repository contains the code and data for the paper *Interpretable Machine Learning Enabled Inorganic Reaction Classification and Synthesis Condition Prediction* by Karpovich et al.

# Installation Instructions
- Clone this repository and navigate to it. 
- Create the conda enivornment for the regression tasks. `conda env create --name regression_env --file requirements_regression.txt`
- Create the conda enivornment for the CVAE tasks. `conda env create --name CVAE_env --file requirements_CVAE.txt`
- Switch to the new environment, depending on which notebook you are running. `conda activate <env_name>`
- Add the environment to jupyter and activate it `python -m ipykernel install --name <env_name>`

# Data
The full datasets used in the paper are available online. Data must be downloaded to an appropriate `data` folder before and preprocessed before any of the notebooks can be run. The data used in this work is from the following papers:
- Kononova, O., Huo, H., He, T., Rong Z., Botari, T., Sun, W., Tshitoyan, V. and Ceder, G. Text-mined dataset of inorganic materials synthesis recipes. Sci Data 6, 203 (2019). (https://doi.org/10.1038/s41597-019-0224-1)
  - Github link to dataset: (https://github.com/CederGroupHub/text-mined-synthesis_public)
- Wang, Z., Kononova, O., Cruse, K., He, T., Huo, H., Fei, Y., Zeng, Y., Sun, Y., Cai, Z., Sun, W. and Ceder, G. Dataset of solution-based inorganic materials synthesis procedures extracted from the scientific literature. Sci Data 9, 231 (2022). (https://doi.org/10.1038/s41597-022-01317-2)
  - Github link to dataset: (https://github.com/CederGroupHub/text-mined-solution-synthesis_public)

# Usage
Each folder pertains to a particular task (synthesis route classification or synthesis condition prediction) containing the associated Jupyter notebooks and python code.
- The `rxn_classification` folder contains the necessary code for the reaction classification tasks.
- The `rxn_condition_prediction` folder contains the necessary code for the reaction conditions prediction tasks.
  - The `CVAE` folder contains the necessary code for training and evaluating the conditional variational autoencoder model for condition prediction.
  - The `regression` folder contains the necessary code for training and evaluating the regression models for condition prediction.

# Cite
If you use this dataset or code in your work please cite as:
```
@article{doi:10.1021/acs.chemmater.2c03010,
author = {Karpovich, Christopher and Pan, Elton and Jensen, Zach and Olivetti, Elsa},
doi = {10.1021/acs.chemmater.2c03010},
journal = {Chemistry of Materials},
number = {3},
pages = {1062--1079},
title = {{Interpretable Machine Learning Enabled Inorganic Reaction Classification and Synthesis Condition Prediction}},
url = {https://doi.org/10.1021/acs.chemmater.2c03010},
volume = {35},
year = {2023}
}
```

# Disclaimer
This is research code shared without support or guarantee of quality. Please report any issues found by opening an issue in this repository. 
