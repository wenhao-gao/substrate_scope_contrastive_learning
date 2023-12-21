# ContraScope

Repo of substrate metric leanring project. 

# Environment

Running `create_env.sh` would create the virtual environment that resolves most of the dependencies in this repo. Note that I am using PyTorch 2.0 and CUDA 11.7, if you are using other versions, please modify the last line accringly (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html for details).

# Model training

To train the model, one can run following command from this directory:

`scipts/pre_train.sh`

To conduct hyper-parameter tuning, one can run:

`scipts/ray_tune.sh`

# Experiment monitoring

I use wandb (https://docs.wandb.ai/) for experiment monitoring. If you want to use wandb to log your results, please login with your wandb account first (see https://docs.wandb.ai/quickstart). If you don't want to use wandb, you can turn it off by using argument `--wandb disabled`.