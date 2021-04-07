# GNN for Dune SIM events

## Setup

(some small packages may be missing)

```
conda create -n mymlenv python=3.8
conda activate mymlenv
```

Depending on your CUDA version, install `torch_geometric` (for older/newer CUDA versions you might need torch 1.6/1.8 - check with `nvidia-smi`):

```
conda install pytorch=1.7 torchvision torchaudio -c pytorch

python -c "import torch; print(torch.version.cuda)"
11.0

CUDA="cu110"

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-geometric
```

Install `unionfind` (https://github.com/eldridgejm/unionfind):

```
conda install -c anaconda cython
pip install git+https://github.com/eldridgejm/unionfind.git
```

Some small helpful packages:

```
pip install tqdm
```

And finally this repo:

```
git clone https://github.com/tklijnsma/duneml.git
```


## Usage

First download and preprocess the data:

```
python dataset.py fromscratch
```

Then train with simply:

```
python train.py
```

