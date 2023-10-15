# RNABERT
This repo contains the code for our paper "Informative RNA-base embedding for functional RNA clustering and structural alignment". Please contact me at akiyama@dna.bio.keio.ac.jp for any question. Please cite this paper if you use our code or system output.

In this package, we provides resources including: source codes of the RNABERT model, pre-trained weights, prediction module.

## 1. Environment setup

The code is written with python Python 3.6.5. Our code requires PyTorch version >= 1.4.0, biopython version >=1.76, and C++17 compatible compiler. Please follow the instructions here: https://github.com/pytorch/pytorch#installation.
Also, please make sure you have at least one NVIDIA GPU. 

#### 1.1 Install the package and other requirements

(Required)

```
cd RNABERT
python setup.py install
```