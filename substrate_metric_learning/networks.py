import torch
from torch_geometric.nn import MLP, GINConv, global_add_pool


def local_pick_pool(x, idx, batch):
    mols_start_from = torch.cat([batch.new_zeros(1), batch.bincount()]).cumsum(0)[:-1]
    idx = idx + mols_start_from.view(-1, 1)
    x = x[idx]
    return x.view(x.shape[0], -1)

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, pool='global'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.pool_method = pool
        
        if pool == 'global':
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = global_add_pool
        elif pool == 'c':
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = local_pick_pool
        elif pool == 'cx':
            self.mlp = MLP([hidden_channels*2, hidden_channels, out_channels],
                        norm=None, dropout=0.5)
            self.pool = local_pick_pool

    def forward(self, x, edge_index, batch, atm_idx=None):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        if atm_idx is not None:
            x = self.pool(x, atm_idx, batch)
        else:
            x = self.pool(x, batch)
        return torch.flatten(self.mlp(x)), x

