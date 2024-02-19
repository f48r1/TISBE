# TISBE
Workflow to develop and evaluate TISBE model for Dev Tox
Mastrolorito F., Togo M.V., Gambacorta N., Trisciuzzi D., Giannuzzi V., Bonifazi F., Liantonio A., Imbrici P., De Luca A., Altomare C.D., Ciriaco F., Nicolotti O., "TISBE: A Public Web Platform for the Consensus-Based Explainable Prediction of Developmental Toxicity". *Chemical Research in Toxicology* (2024).
Reference work paper is available [here](https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00310)

## Requirements - Packages
- python >= 3.10
- pandas, numpy :  for data management
- scipy : for distribution function
- sklearn : for main Machine learning classifiers (SVM, RF, ADA, KNN) and metrics
- xgboost package : for XGB classifier
- rdkit : for chemical managament
- networkx : for community detection
- shap : for Explainability analysis
- moses : for internal diversity calculation
- tqdm : not so required

you can follow the notebooks by numerical order

## How to cite
```
@article{tisbe2024,
author = {Mastrolorito, Fabrizio and Togo, Maria Vittoria and Gambacorta, Nicola and Trisciuzzi, Daniela and Giannuzzi, Viviana and Bonifazi, Fedele and Liantonio, Antonella and Imbrici, Paola and De Luca, Annamaria and Altomare, Cosimo Damiano and Ciriaco, Fulvio and Amoroso, Nicola and Nicolotti, Orazio},
title = {TISBE: A Public Web Platform for the Consensus-Based Explainable Prediction of Developmental Toxicity},
journal = {Chemical Research in Toxicology},
volume = {37},
number = {2},
pages = {323-339},
year = {2024},
doi = {10.1021/acs.chemrestox.3c00310},
URL = {https://doi.org/10.1021/acs.chemrestox.3c00310},
}
```
