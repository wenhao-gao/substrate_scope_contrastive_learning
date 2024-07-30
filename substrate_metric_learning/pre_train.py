import os
import time
import random
import yaml
from flags import FLAGS, parse_flags
from features import smiles_to_graph_substrate, smiles_to_graph
import pandas as pd
import numpy as np
import wandb
import time

import torch
from torch_geometric.loader import DataLoader

from ipdb import set_trace as st
from utils import *
from evaluate import *
from networks import *
from substrate_loss import *
from loader import SC_GTDataLoader

torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(mode=True, warn_only=True)

if __name__ == "__main__":
    parse_flags()
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
    config = yaml.safe_load(open(FLAGS.config_path))
    config = Objdict(config)

    # for key, value in FLAGS.__dict__.items():
    #     if key in config and FLAGS.__dict__[key] is not None:
    #         config[key] = value

    time_start = time.time()

    if ~FLAGS.debug:
        os.environ["WANDB_MODE"] = FLAGS.wandb
        if not FLAGS.wandb_log_code:
            os.environ["WANDB_DISABLE_CODE"] = "false"
        wandb.login()
        run = wandb.init(entity='whgao', project=FLAGS.wandb_project, config=config)

    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    if FLAGS.output_path is None:
        FLAGS.output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", cur_time)
    
    if ~FLAGS.debug:
        if not os.path.exists(FLAGS.output_path):
            os.mkdir(FLAGS.output_path)
        yaml.dump(config, open(os.path.join(FLAGS.output_path, 'config.yaml'), 'w'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if FLAGS.dataset_path[-4:] == '.csv':
        data = pd.read_csv(FLAGS.dataset_path)

        if FLAGS.debug:
            data = data[:1000]

        if config.pool == 'c':
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
        # data = data[:2000]
        FLAGS.epochs = 2
        config.batch_size = 128

    print(f"Using config: {config}")
   
    sample_data = train_dataset[0]
    if config.model == 'gin':
        train_loader = DataLoader(train_dataset, config.batch_size, shuffle=False)
        model = Net(sample_data.x.shape[1], config.hidden_channels, 1, config.num_layers, pool=config.pool).to(device)
    elif config.model == 'gt':
        train_loader = SC_GTDataLoader(train_dataset, config.batch_size, shuffle=False)
        model = GraphTransformer(
            dim=sample_data.x.shape[1],
            depth=config.num_layers,
            dim_head=config.hidden_channels,
            edge_dim=sample_data.edge_attr.shape[1],
            heads=config.num_heads,
            gated_residual=config.gated_residual,
            with_feedforwards=config.with_feedforwards,
            norm_edges=config.norm_edges,
            rel_pos_emb=config.rel_pos_emb,
            accept_adjacency_matrix=config.accept_adjacency_matrix,
            mlp_hidden_channels=config.mlp_hidden_channels,
            out_channels=1,
            pool=config.pool
        ).to(device)
    else:
        raise ValueError('Invalid model type')

    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    torch.save(model.state_dict(), os.path.join(FLAGS.output_path, config.model + f"_epoch_{0}.pth"))

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.momentum, 0.999))
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(config.momentum, 0.999))
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    else:
        raise ValueError('Invalid optimizer type')
    
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

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
        same_halogen_negative=config.same_halogen_negative,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
    )

    r2_max = -9999
    mi_max = 0
    ft_max = -9999

    for epoch in range(1, FLAGS.epochs + 1):

        model.train()
        loss_list = []
        loss_ap_list = []
        loss_an_list = []
        grad_list = []

        for batch_idx, data in enumerate(train_loader):
            if config.model == 'gin':
                data = data.to(device)
                labels = data.scope_id
                values = data.y
                smiles_list = data.smiles
                atm_cls = data.atm_cls
                optimizer.zero_grad()
                out, embeddings = model(data.x, data.edge_index, data.batch, data.atm_idx)
            elif config.model == 'gt':
                nodes, edges, adj_mat, mask, values, labels, atm_idx, smiles_list, atm_cls = data
                nodes, edges, adj_mat, mask, values, labels, atm_idx, atm_cls = \
                    nodes.to(device), edges.to(device), adj_mat.to(device), mask.to(device), values.to(device), labels.to(device), atm_idx.to(device), atm_cls.to(device)
                optimizer.zero_grad()
                embeddings = model(nodes=nodes, edges=edges, adj_mat=adj_mat, mask=mask, atm_idx=atm_idx)[1]
                # Write a loop to iterate through the batch and calculate the embeddings
                # minibatch_size = 32
                # embeddings = torch.zeros((nodes.size(0), sample_data.x.shape[1])).to(device)
                # for i in range(0, nodes.size(0), minibatch_size):
                #     minibatch_embeddings = model(
                #         nodes=nodes[i:i+minibatch_size], 
                #         edges=edges[i:i+minibatch_size], 
                #         adj_mat=adj_mat[i:i+minibatch_size], 
                #         mask=mask[i:i+minibatch_size],
                #         atm_idx=atm_idx[i:i+minibatch_size]
                #     )[1]
                #     embeddings[i:i+minibatch_size] = minibatch_embeddings
                #     torch.cuda.empty_cache()       
            else:
                raise ValueError('Invalid model type')
            loss, loss_ap, loss_an = loss_func(embeddings, values, labels, smiles_list, atm_cls)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()
            loss_list.append(loss.detach().cpu().item() * config.batch_size / len(train_dataset))
            loss_ap_list.append(loss_ap.detach().cpu().item() * config.batch_size / len(train_dataset))
            loss_an_list.append(loss_an.detach().cpu().item() * config.batch_size / len(train_dataset))
            grad_list.append(check_grad(model))

        scheduler_lr.step()
        curr_lr = scheduler_lr.get_lr()
            # print(f"Batch Index: {batch_idx}, Loss: {loss.detach().cpu().item():.4f}, Grad: {check_grad(model):.4f}")

        r2_doyle, mut_info_doyle, r2_hammett, mut_info_hammett, r2_charge, mut_info_charge, r2_nmr, mut_info_nmr, f_test_doyle, f_test_hammett, f_test_charge, f_test_nmr \
            = evaluate(model, device)
        sum_r2 = r2_doyle + r2_hammett + r2_charge + r2_nmr
        sum_mi = mut_info_doyle + mut_info_hammett + mut_info_charge + mut_info_nmr
        sum_ft = f_test_doyle + f_test_hammett + f_test_charge + f_test_nmr
        
        if sum_r2 > r2_max:
            r2_max = sum_r2

        if sum_mi > mi_max:
            mi_max = sum_mi

        if sum_ft > ft_max:
            ft_max = sum_ft
        
        if FLAGS.debug:
            print(f"Epoch: {epoch}, Loss: {sum(loss_list):.4f}, Grad: {sum(grad_list):.4f}, LR: {curr_lr[0]}\
                  R2 Doyle: {r2_doyle:.4f}, Mut_info Doyle: {mut_info_doyle:.4f}, \
                  R2 Hammett: {r2_hammett:.4f}, Mut_info Hammett: {mut_info_hammett:.4f}, \
                  R2 Charge: {r2_charge:.4f}, Mut_info Hammett: {mut_info_charge:.4f}, \
                  R2 NMR: {r2_nmr:.4f}, Mut_info Hammett: {mut_info_nmr:.4f}")
        else:
            wandb.log({
                "epoch": epoch, 
                "loss": sum(loss_list), 
                "loss_ap": sum(loss_ap_list), 
                "loss_an": sum(loss_an_list), 
                "grad": sum(grad_list), 
                "lr": curr_lr[0],
                "r2_doyle": r2_doyle, 
                "mut_info_doyle": mut_info_doyle,
                "f_test_doyle": f_test_doyle,
                "r2_hammett": r2_hammett,
                "mut_info_hammett": mut_info_hammett,
                "f_test_hammett": f_test_hammett,
                "r2_charge": r2_charge, 
                "mut_info_charge": mut_info_charge,
                "f_test_charge": f_test_charge,
                "r2_nmr": r2_nmr,
                "mut_info_nmr": mut_info_nmr,
                "f_test_nmr": f_test_nmr,
                "sum_r2": sum_r2,
                "max_sum_r2": r2_max,
                "sum_mi": sum_mi,
                "max_sum_mi": mi_max,
                "sum_ft": sum_ft,
                "max_sum_ft": ft_max,
                })

            if sum_r2 > 1.5 and epoch > 0:
                torch.save(model.state_dict(), os.path.join(FLAGS.output_path, config.model + f"_epoch_{epoch}_sum_r2_{sum_r2:.3f}.pth"))
            if sum_mi > 2.0 and epoch > 0:
                torch.save(model.state_dict(), os.path.join(FLAGS.output_path, config.model + f"_epoch_{epoch}_sum_mi_{sum_mi:.3f}.pth"))
            if sum_ft > 86.0 and epoch > 0:
                torch.save(model.state_dict(), os.path.join(FLAGS.output_path, config.model + f"_epoch_{epoch}_sum_ft_{sum_ft:.3f}.pth"))

    print(f"Finished training in {time.time() - time_start} seconds")

    if ~FLAGS.debug:
        torch.save(model.state_dict(), os.path.join(FLAGS.output_path, config.model + f"_epoch_{epoch}_sum_r2_{sum_r2:.3f}.pth"))
        run.finish()
