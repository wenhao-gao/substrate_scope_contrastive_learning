# ContraScope

This repository includes all the code for our study: Substrate Scope Contrastive Learning: Repurposing Human Bias to Learn Atomic Representations. 
Additionally, we provide a pre-trained model [here](results/default/gin_epoch_56_sum_r2_1.538.pth) and validation data, which encompasses the 500 most frequently used aryl halides from the CAS Content Collection

# Environment

Executing `create_env.sh` sets up the virtual environment, addressing most dependencies for this repository. Note that we used PyTorch 2.0 and CUDA 11.7 for our experiments. For different versions, adjust the correponding lines as needed, referring to [the official PyG site](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for guidance.

# Usage

To train the model, execute the following command from this directory:

`scipts/pre_train.sh`

For hyper-parameter tuning, run:

`scipts/ray_tune.sh`

For encoding new aryl halides, refer to [this notebook](notebooks/Fig2.ipynb). The necessary code includes (have not tested yet):
```python
from substrate_metric_learning.networks import Net
from substrate_metric_learning.features import smiles_to_graph_substrate

config_path = os.path.join(HOME_DIR, "configs/hparams_default.yaml")
config = Objdict(yaml.safe_load(open(config_path)))
input_dim = 133
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_pretrained = Net(input_dim, config.hidden_channels, 1, config.num_layers, config.pool).to(device)
model_pretrained.load_state_dict(torch.load("results/default/gin_epoch_56_sum_r2_1.538.pth"))

@torch.no_grad()
def get_embedding_from_smi(smi_list, c_index_list, model, device):
    assert len(smi_list) == len(c_index_list)
    assert model.pool_method == 'c'
    train_dataset = [smiles_to_graph_substrate(smiles=smi_list[i], s=0, y=0, atm_idx=[c_index_list[i]]) for i in range(len(smi_list))]
    loader = DataLoader(train_dataset, 128, shuffle=False)
    embeddings = []
    for data in loader:
        data = data.to(device)
        _, emb = model(data.x, data.edge_index, data.batch, data.atm_idx)
        embeddings.append(emb.cpu().detach().numpy())
    return np.concatenate(embeddings, axis=0)
```

# Experiment monitoring

I use [wandb](https://docs.wandb.ai/) for experiment monitoring. If you want to use wandb to log your results, please login with your wandb account first (see https://docs.wandb.ai/quickstart). If you don't want to use wandb, you can turn it off by using argument `--wandb disabled`.

# License
This project is licensed under the terms of the [Apache 2.0 License](LICENSE).

# Contact 
Please contact gaowh19@gmail.com for help or submit an issue. 

<!-- # Cite Us
This package was developed as a spin-off from [our paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/8644353f7d307baaf29bc1e56fe8e0ec-Paper-Datasets_and_Benchmarks.pdf) (NeurIPS 2022). If you find this package useful, please consider citing:

```
@article{gao2022sample,
  title={Sample efficiency matters: a benchmark for practical molecular optimization},
  author={Gao, Wenhao and Fu, Tianfan and Sun, Jimeng and Coley, Connor},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={21342--21357},
  year={2022}
}
``` -->
