# MatRIS
This repository contains the official PyTorch implementation of **MatRIS** (a foundation model for **Mat**erials **R**epresentation and **I**nteraction **S**imulation). 
The code is currently under preparation and will be open-sourced soon.

## MatRIS-v0.5.0-MPtrj
### MPtrj training set MAE
| Energy(mev/atom) | Force(mev/A) | Stress(GPa) | Magmom(muB)|
|:--------:|:--------:|:--------:|:--------:|
|    6.8   |   20.9   |   194.3  |   23.7   |

**Loss prefactor**: e:f:s:m = 5:5:0.1:0.1

**Training time**: 78 GPU-days on A800 

**Model parameters**: 5,825,620

### Matbench-Discovery Benchmark
```
               Full Test       Unique Prototypes      10k Most Stable
F1               0.798               0.809                 0.984
DAF              4.456               5.049                 6.338
Precision        0.765               0.772                 0.969
Recall           0.834               0.850                 1.000
Accuracy         0.927               0.938                 0.969
TPR              0.834               0.850                 1.000
FPR              0.053               0.046                 1.000
TNR              0.947               0.954                 0.000
FNR              0.166               0.150                 0.000
TP           36755.000           28379.000              9693.000
FP           11312.000            8391.000               307.000
TN          201559.000          173723.000                 0.000
FN            7337.000            4995.000                 0.000
MAE              0.035               0.037                 0.026
RMSE             0.080               0.082                 0.057
R2               0.800               0.803                 0.926
```

###  Heat-Conductivity Benchmark (K_SRME: 0.861)

## MatRIS-v1.0.0-MPtrj
This version will be released soon.
