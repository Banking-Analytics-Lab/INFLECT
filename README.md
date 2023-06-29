# INFLECT
Code for the paper Tiukhova et al. (2023) INFLECT-DGNN: INFLuencer prEdiCTion with Dynamic Graph Neural Networks

## Project structure: 

The project repo holds the following structure
```
 |-models
 | |-GNNs.py
 | |-RNNs.py
 | |-decoder.py
 | |-models.py
 |-utils
 | |-utils.py
 |-train.py
 |-requirements.txt
  
 

```
### models

This folder contains the .py files used to make combinations of encoder and decoder in dynamic GNN models as well as create baseline models.

### utils

This folder contains a .py file that provides functions for several files.

### make_data.py

The script to generate the network data and preprocess it. 

### train.py

The script to run the experiments. 

### requirements.txt

The files that lists all of a project's dependencies.
