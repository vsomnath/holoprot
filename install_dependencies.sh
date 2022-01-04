#!/bin/bash

source $(conda info --root)/etc/profile.d/conda.sh

conda create -y --name prot python=3.7.3
conda activate prot
conda env update --file $PWD/environment.yml

CUDA=cu101
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-geometric

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    pip install "https://github.com/PyMesh/PyMesh/releases/download/v0.3/pymesh2-0.3-cp37-cp37m-linux_x86_64.whl"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    pip install "https://github.com/PyMesh/PyMesh/releases/download/v0.3/pymesh2-0.3-cp37-cp37m-macosx_10_15_x86_64.whl"
else
    echo "$OSTYPE"
fi

python setup.py develop
