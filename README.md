# EILS

This is the package version for ANU Honours project -- Fast Parameter-free Clustering using Enhanced Iterative Label Spreading.
[![DOI](https://zenodo.org/badge/462578309.svg)](https://zenodo.org/badge/latestdoi/462578309)




## ILS Versions

- Original version: in [ils/basic_ils.py](https://github.com/Thea-Hsu/EILS/blob/main/ils/basic_ils.py)
  - Note: Improvements have been made to the outdated pandas.append() method.
- Enhance version: this repository
- Scikit-Learn version: https://github.com/ajp1717/Summer-Project-Advanced-and-Interpretable-Unsupervised-Learning



## Install Environment

```
conda env create --file environment.yml -n EILS
conda activate EILS	
```

OR

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
git clone https://github.com/Thea-Hsu/EILS.git
cd EILS
poetry init
```



## Package Introduction

This package designs an enhanced version of ILS clustering.



## Package Usage

For details, please see the demo.ipynb

- [cfsdp_demo.ipynb](https://github.com/Thea-Hsu/EILS/blob/main/cfsdp_demo.ipynb)
- [dirchlet_AND_agglomerative_demo.ipynb](https://github.com/Thea-Hsu/EILS/blob/main/dirchlet_AND_agglomerative_demo.ipynb)



## TODO

+ Add explainations for demo files, it is expected to be completed after the thesis.



### Authors

- [Xinyue (Thea) Xu](https://github.com/Thea-Hsu)
- [Dr. Amanda Parker](https://github.com/ajp1717): supervisor, designed the original version of Iterative Label Spreading.

### Other Contributors
- Jon Ting: for testing the usage of ILS and EILS.
