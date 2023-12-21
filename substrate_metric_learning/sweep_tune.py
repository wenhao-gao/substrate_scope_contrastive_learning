
import numpy as np
from flags import FLAGS, parse_flags
import random
import yaml
from features import smiles_to_graph_substrate, smiles_to_graph
import pandas as pd
import wandb
import torch
from torch_geometric.loader import DataLoader

from ipdb import set_trace as st
from utils import *
from evaluate import *
from networks import *
from substrate_loss import *

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(mode=True, warn_only=True)

if __name__ == "__main__":

    parse_flags()
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
    hparam_default = yaml.safe_load(open(FLAGS.config_path))
    hparam_space = yaml.safe_load(open(FLAGS.tune_config_path))
    wandb.login()

    def _func():
        with wandb.init(config=hparam_default) as run:
            config = wandb.config

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if FLAGS.dataset_path[-4:] == '.csv':
                data = pd.read_csv(FLAGS.dataset_path)
                if config.pool == 'c':
                    train_dataset = [smiles_to_graph_substrate(smiles=data['Arylhalide'][ind], 
                                                            s=data['ScopeID'][ind], 
                                                            y=data['Yield'][ind]/100, 
                                                            atm_idx=[data['aroma_c_index'][ind]]) for ind in range(len(data))]
                elif config.pool == 'cx':
                    train_dataset = [smiles_to_graph_substrate(smiles=data['Arylhalide'][ind], 
                                                            s=data['ScopeID'][ind], 
                                                            y=data['Yield'][ind]/100, 
                                                            atm_idx=[data['aroma_c_index'][ind], data['halide_index'][ind]]) for ind in range(len(data))]
                elif config.pool == 'global':
                    train_dataset = [smiles_to_graph_substrate(smiles=data['Arylhalide'][ind], 
                                                            s=data['ScopeID'][ind], 
                                                            y=data['Yield'][ind]/100) for ind in range(len(data))]
                else:
                    raise ValueError('Invalid pool type')
            elif FLAGS.dataset_path[-4:] == '.ptg':
                train_dataset = torch.load(FLAGS.dataset_path)
            else:
                raise ValueError('Invalid dataset path')
            
            if FLAGS.debug:
                FLAGS.epochs = 2
            
            train_loader = DataLoader(train_dataset, config.batch_size, shuffle=False)

            data = train_dataset[0]
            model_pretrained = Net(data.x.shape[1], config.hidden_channels, 1, config.num_layers, pool=config.pool).to(device)
            for m in model_pretrained.modules():
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

            optimizer = torch.optim.Adam(model_pretrained.parameters(), lr=config.lr)

            swap = False

            if config.distance == 'l2':
                from pytorch_metric_learning.distances import LpDistance
                distance_class = LpDistance()
            elif config.distance == 'l1':
                from pytorch_metric_learning.distances import LpDistance
                distance_class = LpDistance(normalize_embeddings=False, p=1)
            elif config.distance == 'cosine':
                from pytorch_metric_learning.distances import CosineSimilarity
                distance_class = CosineSimilarity()
                swap = True
            elif config.distance == 'snr':
                from pytorch_metric_learning.distances import SNRDistance
                distance_class = SNRDistance()
            else:
                raise ValueError('Invalid distance type')
            
            loss_func = SubstrateTripletLoss(
                device=device,
                margin=config.margin,
                smooth_loss=config.smooth_loss,
                triplets_per_anchor=config.triplets_per_anchor,
                distance=distance_class,
                swap=swap,
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma,
            )

            r2_max = -9999

            for epoch in range(1, FLAGS.epochs + 1):

                model_pretrained.train()
                loss_list = []
                loss_ap_list = []
                loss_an_list = []
                grad_list = []
                for batch_idx, data in enumerate(train_loader):
                    data = data.to(device)
                    labels = data.scope_id
                    values = data.y
                    smiles_list = data.smiles
                    optimizer.zero_grad()
                    out, embeddings = model_pretrained(data.x, data.edge_index, data.batch, data.atm_idx)
                    loss, loss_ap, loss_an = loss_func(embeddings, values, labels, smiles_list)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model_pretrained.parameters(), max_norm=config.max_norm)
                    optimizer.step()
                    loss_list.append(loss.detach().cpu().item())
                    loss_ap_list.append(loss_ap.detach().cpu().item())
                    loss_an_list.append(loss_an.detach().cpu().item())
                    grad_list.append(check_grad(model_pretrained))

                test_r2, mut_info = test_on_doyle(model_pretrained, device)

                if test_r2 > r2_max:
                    r2_max = test_r2

                wandb.log({
                    "epoch": epoch, 
                    "loss": sum(loss_list), 
                    "loss_ap": sum(loss_ap_list), 
                    "loss_an": sum(loss_an_list), 
                    "grad": sum(grad_list), 
                    "test_r2": test_r2, 
                    "mut_info": mut_info,
                    "max_r2": r2_max
                    })

    sweep_id = wandb.sweep(hparam_space)
    wandb.agent(sweep_id, function=_func, count=FLAGS.count, entity='whgao')
