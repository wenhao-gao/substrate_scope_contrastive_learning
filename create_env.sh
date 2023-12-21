#!/bin/bash

conda env create -f molfit_env.yml
conda activate molfit
pip install -U "ray"
pip install "ray[tune]"
pip install torch torchvision torchaudio torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
