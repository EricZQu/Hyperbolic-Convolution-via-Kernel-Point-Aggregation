# Hyperbolic Convolution via Kernel Point Aggregation

This repo provides the official implementation of the HKConv from the following paper.

>**Hyperbolic Convolution via Kernel Point Aggregation**  
Eric Qu, Dongmian Zou  
arXiv: [https://arxiv.org/abs/2306.08862](https://arxiv.org/abs/2306.08862)

### Environment

The code is tested on Python 3.9, PyTorch 1.12.1, and CUDA 11.3.

First, install PyTorch from [https://pytorch.org/](https://pytorch.org/). Then, install the other dependencies by

```bash
pip install -r requirements.txt
```

### Training

The training scripts are in `scripts\`. You can use them by

```bash
bash scripts/experiment/dataset.experiment.sh
```

### File Descriptions

`data/`: Datasets  
`distribution/`: Hyperbolic Distributions  
`kernels/`: Kernel generation  
`layers/`: Hyperbolic layers  
`manifolds/`: Manifold calculations  
`models/`: GNN models  
`optim/`: Optimization on manifolds  
`scripts/`: Training scripts  
`utils/`: Utility files  
`train.py`: Training scripts


Our HKConv is located in `layers/hyp_layers.py`.
