# MatRIS
This repository contains the official PyTorch implementation of **MatRIS** (a foundation model for **Mat**erials **R**epresentation and **I**nteraction **S**imulation). 
The code is currently under preparation and will be open-sourced soon.

## MPTrj Result (MatRIS-v0.5.0-MPtrj)
### MPtrj training set MAE
| Energy(mev/atom) | Force(mev/A) | Stress(GPA) | Magmom(muB)|
|:--------:|:--------:|:--------:|:--------:|
|    6.8   |   20.9   |   194.3  |   23.7   |

**Loss prefactor**: e:f:s:m = 5:5:0.1:0.1

**Training time**: 78 GPU-days on A800 

**Model parameters**: 5,825,620

## Matbench-Discovery Benchmark (MatRIS-v0.5.0-MPtrj)
```
               Full Test       Unique Prototypes      10k Most Stable
F1               0.798               0.809                 0.984
DAF              4.457               5.049                 6.338
Precision        0.765               0.772                 0.969
Recall           0.834               0.850                 1.000
Accuracy         0.928               0.938                 0.969
TPR              0.834               0.850                 1.000
FPR              0.053               0.046                 1.000
TNR              0.947               0.954                 0.000
FNR              0.166               0.150                 0.000
TP           36767.000           28384.000              9689.000
FP           11304.000            8388.000               311.000
TN          201567.000          173726.000                 0.000
FN            7325.000            4990.000                 0.000
MAE              0.035               0.037                 0.026
RMSE             0.080               0.082                 0.057
R2               0.801               0.804                 0.926
```

##  Heat-Conductivity Benchmark (K_SRME: 0.865)
