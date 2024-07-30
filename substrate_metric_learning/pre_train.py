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
   
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=False)

    data = train_dataset[0]
    model_pretrained = Net(data.x.shape[1], config.hidden_channels, 1, config.num_layers, pool=config.pool).to(device)
    for m in model_pretrained.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    torch.save(model_pretrained.state_dict(), os.path.join(FLAGS.output_path, f"gin_epoch_{0}.pth"))

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_pretrained.parameters(), lr=config.lr, betas=(config.momentum, 0.999))
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model_pretrained.parameters(), lr=config.lr, momentum=config.momentum)
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
            atm_cls = data.atm_cls
            optimizer.zero_grad()
            out, embeddings = model_pretrained(data.x, data.edge_index, data.batch, data.atm_idx)
            loss, loss_ap, loss_an = loss_func(embeddings, values, labels, smiles_list, atm_cls)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_pretrained.parameters(), max_norm=config.max_norm)
            optimizer.step()
            loss_list.append(loss.detach().cpu().item())
            loss_ap_list.append(loss_ap.detach().cpu().item())
            loss_an_list.append(loss_an.detach().cpu().item())
            grad_list.append(check_grad(model_pretrained))
        scheduler_lr.step()
            # print(f"Batch Index: {batch_idx}, Loss: {loss.detach().cpu().item():.4f}, Grad: {check_grad(model_pretrained):.4f}")

        r2_doyle, mut_info_doyle, r2_hammett, mut_info_hammett, r2_charge, mut_info_charge, r2_nmr, mut_info_nmr, f_test_doyle, f_test_hammett, f_test_charge, f_test_nmr \
            = evaluate(model_pretrained, device)
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
            print(f"Epoch: {epoch}, Loss: {sum(loss_list):.4f}, Grad: {sum(grad_list):.4f}, \
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
                torch.save(model_pretrained.state_dict(), os.path.join(FLAGS.output_path, f"gin_epoch_{epoch}_sum_r2_{sum_r2:.3f}.pth"))
            if sum_mi > 2.0 and epoch > 0:
                torch.save(model_pretrained.state_dict(), os.path.join(FLAGS.output_path, f"gin_epoch_{epoch}_sum_mi_{sum_mi:.3f}.pth"))
            if sum_ft > 86.0 and epoch > 0:
                torch.save(model_pretrained.state_dict(), os.path.join(FLAGS.output_path, f"gin_epoch_{epoch}_sum_ft_{sum_ft:.3f}.pth"))

    print(f"Finished training in {time.time() - time_start} seconds")

    if ~FLAGS.debug:
        torch.save(model_pretrained.state_dict(), os.path.join(FLAGS.output_path, f"gin_epoch_{epoch}_sum_r2_{sum_r2:.3f}.pth"))
        run.finish()
