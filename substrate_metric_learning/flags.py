import argparse
from ipdb import set_trace as st

FLAGS = argparse.Namespace() 

def parse_additonal_params(additonal_params_list):
    add_params = {}
    for i in range(0, len(additonal_params_list), 2):
        add_params[additonal_params_list[i][2:]] = additonal_params_list[i+1]
    return add_params

def parse_flags() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--tune_config_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--search_alg", type=str, default="random", choices=["random", "bayes_ucb", "bayes_ei", "bayes_poi"])

    parser.add_argument('--wandb', type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument('--wandb_log_code', action='store_true')
    parser.add_argument("--wandb_project", type=str, default="substrate_metric_learning")

    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_norm", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument("--smooth_loss", action='store_true')
    parser.add_argument("--triplets_per_anchor", type=int, default=None)
    parser.add_argument("--distance", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)

    args, additonal_params = parser.parse_known_args()
    # args = parser.parse_args()
    additonal_params = parse_additonal_params(additonal_params)
    FLAGS.__dict__.update(args.__dict__)
    FLAGS.__dict__.update(additonal_params)
