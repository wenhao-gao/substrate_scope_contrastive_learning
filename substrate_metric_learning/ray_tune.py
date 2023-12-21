
import numpy as np
from flags import FLAGS, parse_flags
import random
import yaml
from features import smiles_to_graph_substrate, smiles_to_graph
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from ipdb import set_trace as st
from utils import *
from evaluate import *
from networks import *
from substrate_loss import *

from ray import tune
from ray.air import session
from ray.air.integrations.wandb import setup_wandb
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch

torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(mode=True, warn_only=True)

if __name__ == "__main__":

    parse_flags()
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
    hparam_default = yaml.safe_load(open(FLAGS.config_path))

    def train_function_wandb(config):

        torch.use_deterministic_algorithms(mode=True, warn_only=True)
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(FLAGS.seed)

        config = Objdict(config)
        wandb = setup_wandb(config)

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
            FLAGS.epochs = 4
        
        print(f"Using config: {config}")
        
        train_loader = DataLoader(train_dataset, config.batch_size, shuffle=False)

        data = train_dataset[0]
        model_pretrained = Net(data.x.shape[1], config.hidden_channels, 1, config.num_layers, pool=config.pool).to(device)
        for m in model_pretrained.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

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
                optimizer.zero_grad()
                out, embeddings = model_pretrained(data.x, data.edge_index, data.batch, data.atm_idx)
                loss, loss_ap, loss_an = loss_func(embeddings, values, labels, smiles_list)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model_pretrained.parameters(), max_norm=config["max_norm"])
                optimizer.step()
                loss_list.append(loss.detach().cpu().item())
                loss_ap_list.append(loss_ap.detach().cpu().item())
                loss_an_list.append(loss_an.detach().cpu().item())
                grad_list.append(check_grad(model_pretrained))
            scheduler_lr.step()

            r2_doyle, mut_info_doyle, r2_hammett, mut_info_hammett, r2_charge, mut_info_charge, r2_nmr, mut_info_nmr, f_test_doyle, f_test_hammett, f_test_charge, f_test_nmr \
                = evaluate(model_pretrained, device)
            sum_r2 = np.minimum(r2_doyle, 0.5) + r2_hammett + r2_charge + r2_nmr
            sum_mi = mut_info_doyle + mut_info_hammett + mut_info_charge + mut_info_nmr
            sum_ft = f_test_doyle + f_test_hammett + f_test_charge + f_test_nmr

            if sum_r2 > r2_max:
                r2_max = sum_r2

            if sum_mi > mi_max:
                mi_max = sum_mi

            if sum_ft > ft_max:
                ft_max = sum_ft

            session.report({
                "max_sum_r2": r2_max,
                "max_sum_mi": mi_max,
                "max_sum_ft": ft_max,
                })

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

    # hparam_tune = {
    #     "distance": tune.choice(["l1", "snr"]),
    #     "triplets_per_anchor": tune.choice([4, 8, 16, 32]),
    #     "smooth_loss": tune.choice([True, False]),
    #     "beta": tune.uniform(0, 1),
    #     "gamma": tune.uniform(1.0, 5.0),
    #     "margin": tune.uniform(0.1, 2.0),
    #     "wandb": {
    #         "project": "substrate_metric_learning_tune_1"
    #     },
    # }

    # hparam_tune = {
    #     "hidden_channels": tune.choice([4, 8, 16, 32, 64, 128]),
    #     "num_layers": tune.choice([3, 5, 7, 9, 11]),
    #     "batch_size": tune.choice([2048, 4096, 8192]),
    #     "max_norm": tune.choice([1.0, 5.0, 10.0]),
        # "optimizer": tune.choice(["adam", "sgd"]),
        # "momentum": tune.uniform(0.0, 1.0),
        # "lr": tune.loguniform(1e-6, 1e-4),
        # "lr_decay": tune.choice([0.9, 0.99, 0.999, 1.0]),
    #     "wandb": {
    #         "project": "substrate_metric_learning_tune_2"
    #     },
    # }

    # hparam_tune = {
    #     "num_layers": tune.choice([5, 6, 7, 8, 9]),
    #     "batch_size": tune.choice([2048, 4096]),
    #     "triplets_per_anchor": tune.choice([8, 16, 32, 64]),
    #     "smooth_loss": tune.choice([True, False]),
    #     "wandb": {
    #         "project": "substrate_metric_learning_tune_5"
    #     },
    # }
    
    # hparam_tune = {
    #     "num_layers": tune.choice([5, 6, 7, 8, 9]),
    #     "batch_size": tune.choice([2048, 4096]),
    #     "triplets_per_anchor": tune.choice([16, 32]),
    #     "optimizer": tune.choice(['adam', 'sgd']),
    #     "lr": tune.loguniform(1e-6, 1e-4),
    #     "lr_decay": tune.choice([0.9, 0.99, 0.999, 1.0]),
    #     "wandb": {
    #         "project": "substrate_metric_learning_tune_6"
    #     },
    # }

    # hparam_tune = {
    #     "beta": tune.uniform(0, 1),
    #     "gamma": tune.uniform(1.0, 5.0),
    #     "margin": tune.uniform(0.1, 3.0),
    #     "wandb": {
    #         "project": "substrate_metric_learning_tune_7"
    #     },
    # }

    # hparam_tune = {
    #     "beta": tune.uniform(0, 1),
    #     "gamma": tune.randn(4.0, 0.5),
    #     "margin": tune.randn(2.0, 0.5),
    #     "wandb": {
    #         "project": "substrate_metric_learning_tune_8"
    #     },
    # }

    # hparam_tune = {
    #     "momentum": tune.choice([0.8, 0.9, 0.99]),
    #     "lr": tune.loguniform(1e-5, 1e-4),
    #     "lr_decay": tune.choice([0.99, 0.999, 1.0]),
    #     "wandb": {
    #         "project": "substrate_metric_learning_tune_9"
    #     },
    # }

    hparam_tune = {
        "hidden_channels": tune.choice([1, 24, 8, 16, 32, 64]),
        "num_layers": tune.choice([5, 7, 9]),
        "batch_size": tune.choice([2048, 4096]),
        "optimizer": tune.choice(["adam", "sgd"]),
        "momentum": tune.choice([0.8, 0.9, 0.99]),
        "lr": tune.loguniform(1e-5, 1e-4),
        "lr_decay": tune.choice([0.99, 0.999, 1.0]),
        "triplets_per_anchor": tune.choice([4, 8, 16, 32]),
        "beta": tune.uniform(0, 1),
        "gamma": tune.randn(4.0, 1.5),
        "margin": tune.randn(2.0, 1.0),
        "wandb": {
            "project": "substrate_metric_learning_tune_11"
        },
    }

    search_space = hparam_default
    for key in hparam_tune.keys():
        search_space[key] = hparam_tune[key]

    if FLAGS.search_alg == "random":
        alg = None
    elif FLAGS.search_alg == "bayes_ucb":
        """
        Kappa: Parameter to indicate how closed are the next parameters sampled.
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is
            the highest.
        """
        alg = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
        alg = ConcurrencyLimiter(alg, max_concurrent=FLAGS.max_concurrent)
    elif FLAGS.search_alg == "bayes_ei":
        alg = BayesOptSearch(utility_kwargs={"kind": "ei", "kappa": 2.5, "xi": 0.0})
        alg = ConcurrencyLimiter(alg, max_concurrent=FLAGS.max_concurrent)
    elif FLAGS.search_alg == "bayes_poi":
        alg = BayesOptSearch(utility_kwargs={"kind": "poi", "kappa": 2.5, "xi": 0.0})
        alg = ConcurrencyLimiter(alg, max_concurrent=FLAGS.max_concurrent)
    else:
        raise ValueError("Invalid search algorithm.")

    trainable_with_resources = tune.with_resources(train_function_wandb, {"cpu": 12, "gpu": 1})

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="max_sum_r2",
            mode="max",
            search_alg=alg,
            num_samples=FLAGS.count,
        )
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
