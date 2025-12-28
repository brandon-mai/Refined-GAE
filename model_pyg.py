import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, LGConv

class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=2, res=False, norm=False, scale=False, act='relu'):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res
        self.scale = scale
        if scale:
            self.scale_norm = nn.LayerNorm(h_feats)
        self.norm = norm
        if norm:
            self.norms = torch.nn.ModuleList()
            for _ in range(layer - 1):
                self.norms.append(nn.LayerNorm(h_feats))
        if act == 'relu':
            self.act = F.relu
        elif act == 'gelu':
            self.act = F.gelu
        elif act == 'silu':
            self.act = F.silu
        else:
            raise ValueError('Activation function not supported')
        
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.scale:
            self.scale_norm.reset_parameters()
        if self.norm:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.scale:
            x = self.scale_norm(x)
        ori = x
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.res:
                x = x + ori
            if self.norm:
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()

class DotPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_i, x_j):
        x = (x_i * x_j).sum(dim=-1)
        return x.squeeze()

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.2,
                 norm=False, tailact=False, norm_affine=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.Sequential()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        if norm:
            self.lins.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
        self.lins.append(nn.ReLU())
        if dropout > 0:
            self.lins.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if norm:
                self.lins.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
            self.lins.append(nn.ReLU())
            if dropout > 0:
                self.lins.append(nn.Dropout(dropout))
        out_channels = hidden_channels
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        if tailact:
            self.lins.append(nn.LayerNorm(out_channels, elementwise_affine=norm_affine))
            self.lins.append(nn.ReLU())
            self.lins.append(nn.Dropout(dropout))

    def forward(self, x):
        x = self.lins(x)
        return x.squeeze()

class GCN_with_feature(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0, prop_step=2, dropout=0.2, residual=0, concat_skip=False, relu=False, linear=False, conv='GCN'):
        super(GCN_with_feature, self).__init__()
        self.conv_type = conv
        if conv == 'GCN':
            self.conv1 = GCNConv(in_feats, h_feats)
            self.conv2 = GCNConv(h_feats, h_feats)
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, aggr='mean')
            self.conv2 = SAGEConv(h_feats, h_feats, aggr='mean')
        elif conv == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats // 4, heads=4)
            self.conv2 = GATConv(h_feats, h_feats // 4, heads=4)
        elif conv == 'GIN':
            self.mlp1 = MLP(in_feats, h_feats, 2, dropout)
            self.mlp2 = MLP(h_feats, h_feats, 2, dropout)
            self.conv1 = GINConv(self.mlp1, train_eps=True) # train_eps=True roughly matches some GIN variants, checking default. DGL default is learn_eps=False ('mean' aggregator is manual in DGL GIN usually?)
            # In old model.py: GINConv(mlp, 'mean'). PyG GINConv doesn't support 'mean' aggr directly in constructor in older versions, but current GINConv takes aggr='add'.
            # However, for exact match, PyG GINConv(nn, train_eps=False) is sum aggregation.
            # DGL 'mean' aggregator is significant. PyG GIN usually assumes sum.
            # We can use GINConv(nn, aggr='mean').
            self.conv1 = GINConv(self.mlp1, train_eps=False, aggr='mean')
            self.conv2 = GINConv(self.mlp2, train_eps=False, aggr='mean')

        self.prop_step = prop_step
        self.residual = residual
        self.relu = relu
        self.norm = norm
        self.linear = linear
        self.in_feats = in_feats
        self.h_feats = h_feats
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout)
        if self.linear:
            self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, dropout)])
            for _ in range(prop_step - 1):
                self.mlps.append(MLP(h_feats, h_feats, 2, dropout))
        self.concat_skip = concat_skip
        if concat_skip:
            self.post_concat = nn.ModuleList([nn.Linear(h_feats * 2, h_feats) for _ in range(prop_step)])

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x

    def forward(self, edge_index, in_feat, edge_weight=None):
        # NOTE: DGL forward(g, in_feat, e_feat)
        # PyG forward(x, edge_index, edge_weight)
        
        # PyG GCNConv args: x, edge_index, edge_weight
        # PyG SAGEConv args: x, edge_index
        # PyG GATConv args: x, edge_index
        # PyG GINConv args: x, edge_index
        
        # We need to handle edge_weight for GCN.
        
        if self.conv_type == 'GCN':
            h = self.conv1(in_feat, edge_index, edge_weight=edge_weight)
        else:
            h = self.conv1(in_feat, edge_index)
            
        # .flatten(1) is used in DGL because result might be (N, 1, D) for GAT?
        # PyG GATConv output is (N, H*Heads).
        # In DGL GAT: (N, num_heads, out_feats). Flatten(1) makes it (N, num_heads*out_feats).
        # PyG GATConv automatically returns (N, num_heads*out_feats) if created with concat=True (default).
        # So no explicit flatten needed usually, but let's check shapes.
        
        ori = h
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            if self.linear:
                h = self.mlps[i](h)

            if self.conv_type == 'GCN':
                h = self.conv2(h, edge_index, edge_weight=edge_weight)
            else:
                h = self.conv2(h, edge_index)
            
            if self.concat_skip:
                h = self.post_concat[i](torch.cat((h, ori), dim=1))
            else:
                h = h + self.residual * ori
        return h

class LightGCN(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout=0.2, alpha=0.5, exp=False, relu=False, norm=False, conv='GCN'):
        super(LightGCN, self).__init__()
        self.conv_type = conv
        if conv == 'GCN':
            # weight=True in DGL means learnable transformation -> GCNConv
            self.conv1 = GCNConv(in_feats, h_feats, bias=False)
            # weight=False in DGL means NO learnable transformation -> LGConv or GCNConv with fixed identity weights
            # But LGConv is specifically for LightGCN (normalized sum).
            # DGL GraphConv(weight=False) is effectively LGConv: D^-0.5 A D^-0.5 X
            self.conv2 = LGConv() 
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, aggr='mean', bias=False)
            self.conv2 = SAGEConv(h_feats, h_feats, aggr='mean', bias=False)
        elif conv == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats // 4, heads=4)
            self.conv2 = GATConv(h_feats, h_feats // 4, heads=4)
        elif conv == 'GIN':
            self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, dropout)])
            self.convs = nn.ModuleList([GINConv(self.mlps[0], aggr='sum')]) # DGL LightGCN GIN uses aggr='sum' in model.py (line 470)
            for i in range(prop_step - 1):
                self.mlps.append(MLP(h_feats, h_feats, 2, dropout))
                self.convs.append(GINConv(self.mlps[i + 1], aggr='sum'))

        self.prop_step = prop_step
        self.relu = relu
        self.alpha = alpha
        if exp:
            self.alphas = nn.Parameter(alpha ** torch.arange(prop_step))
        else:
            self.alphas = nn.Parameter(torch.ones(prop_step))
        self.norm = norm
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout)

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x

    def forward(self, edge_index, in_feat, edge_weight=None):
        if self.conv_type == 'GIN':
            alpha = F.softmax(self.alphas, dim=0)
            h = self.convs[0](in_feat, edge_index)
            res = h * alpha[0]
            for i in range(1, self.prop_step):
                h = self._apply_norm_and_activation(h, i)
                h = self.convs[i](h, edge_index)
                res += h * alpha[i]
            return res
        
        # GCN/Other Logic
        alpha = F.softmax(self.alphas, dim=0)
        
        # First layer (potentially with weights)
        if self.conv_type == 'GCN':
            h = self.conv1(in_feat, edge_index, edge_weight=edge_weight)
        else:
            h = self.conv1(in_feat, edge_index)

        res = h * alpha[0]
        
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            # Second layer (propagation only for GCN)
            if self.conv_type == 'GCN':
                # LGConv just takes x, edge_index. No edge_weight in standard LGConv signature in some versions but yes in others.
                # Checking PyG docs: LGConv forward(x, edge_index, edge_weight=None)
                h = self.conv2(h, edge_index, edge_weight=edge_weight) 
            elif self.conv_type == 'SAGE': # SAGE doesn't have same "weight=False" easily unless we define it that way.
                # Assuming conv2 corresponds to initialized conv2
                h = self.conv2(h, edge_index)
            elif self.conv_type == 'GAT':
                 h = self.conv2(h, edge_index)

            res += h * alpha[i]
        return res

