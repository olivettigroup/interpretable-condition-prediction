# Interpretable Machine Learning Enabled Inorganic Reaction Classification and Synthesis Condition Prediction
This repository contains the code and data for the paper *Interpretable Machine Learning Enabled Inorganic Reaction Classification and Synthesis Condition Prediction*.

# Installation Instructions
- Clone this repository and navigate to it. 
- Create the conda enivornment. `conda env create -f env/environment.yml`
- Switch to the new environment. `conda activate osda_env`
- Run the setup file. `python setup.py install`
- Add the environment to jupyter `python -m ipykernel install --name osda_env --display-name "osda_gen"`

# Data
The full datasets used in the paper are available online:
- Kononova, O., Huo, H., He, T., Rong Z., Botari, T., Sun, W., Tshitoyan, V. and Ceder, G. Text-mined dataset of inorganic materials synthesis recipes. Sci Data 6, 203 (2019). (https://doi.org/10.1038/s41597-019-0224-1)
  - Github link to dataset: (https://github.com/CederGroupHub/text-mined-synthesis_public)
- Wang, Z., Kononova, O., Cruse, K., He, T., Huo, H., Fei, Y., Zeng, Y., Sun, Y., Cai, Z., Sun, W. and Ceder, G. Dataset of solution-based inorganic materials synthesis procedures extracted from the scientific literature. Sci Data 9, 231 (2022). (https://doi.org/10.1038/s41597-022-01317-2)
  - Github link to dataset: (https://github.com/CederGroupHub/text-mined-solution-synthesis_public)

# Usage
Each folder pertains to a particular task (synthesis route classification or synthesis condition prediction) containing the associated Jupyter notebooks and python code.

# Issues
Please report any issues found by opening an issue in this repository. 
# Cite
If you use this dataset or code in your work please cite as:
```
TBA
```
