import argparse
import itertools
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import add_self_loops, negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np

from loss import auc_loss, hinge_auc_loss, log_rank_loss, pull_loss
from model_pyg import GCN_with_feature, DotPredictor, LightGCN, Hadamard_MLPPredictor, MLP

def parse():
    parser = argparse.ArgumentParser()
    # General Setup
    parser.add_argument("--dataset", default='ogbl-collab', choices=['ogbl-collab'], type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--run_name', default=None, type=str)
    parser.add_argument('--hf_token', default=None, type=str)
    parser.add_argument('--hf_repo_id', default=None, type=str)

    # Training Configuration
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=8192, type=int)
    parser.add_argument("--metric", default='hits@20', type=str)
    parser.add_argument("--loss", default='bce', choices=['bce', 'auc', 'hauc', 'rank', 'pull'], type=str)
    parser.add_argument("--interval", default=100, type=int)
    parser.add_argument("--step_lr_decay", action='store_true', default=True)
    parser.add_argument('--clip_norm', default=1.0, type=float)

    # Data / Sampling
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--maskinput", action='store_true')
    parser.add_argument("--use_valid_as_input", action='store_true')
    parser.add_argument("--pull_interval", default=200, type=int)
    parser.add_argument("--pull_r", default=0.05, type=float)

    # Encoder (GCN / Embedding)
    parser.add_argument("--model", default='GCN', type=str)
    parser.add_argument("--conv", default='GCN', type=str)
    parser.add_argument("--emb_hidden", default=64, type=int)
    parser.add_argument("--hidden", default=64, type=int)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--norm", action='store_true', default=False)
    parser.add_argument("--dp4norm", default=0, type=float) # unused
    parser.add_argument("--dpe", default=0, type=float) # unused
    parser.add_argument("--drop_edge", action='store_true', default=False) # unused
    parser.add_argument("--residual", default=0, type=float)
    parser.add_argument('--linear', action='store_true', default=False)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--exp', action='store_true', default=False)
    parser.add_argument('--init', default='orthogonal', type=str)
    parser.add_argument("--relu", action='store_true', default=False)
    parser.add_argument('--concat_skip', action='store_true', default=False)

    # Decoder (Predictor)
    parser.add_argument("--pred", default='mlp', type=str)
    parser.add_argument("--mlp_layers", default=2, type=int)
    parser.add_argument("--res", action='store_true', default=False)
    parser.add_argument('--scale', action='store_true', default=False)

    args = parser.parse_args()
    return args

args = parse()
print(args)

if args.run_name:
    wandb.init(project='Refined-GAE-PyG', name=args.run_name, config=args)
else:
    wandb.init(project='Refined-GAE-PyG', config=args)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

def eval_hits(y_pred_pos, y_pred_neg, K):
    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}
    
    # y_pred_neg is usually [N_pos * num_neg] or just a large pool
    # The original code logic: "For each positive target node, the negative target nodes are the same."
    # Wait, in the evaluation function of collab.py, neg_valid_edge is used.
    # We should stick to the ogb Evaluator if possible or replicate the logic accurately.
    # The original eval_hits assumes y_pred_neg is a single tensor compared against ALL positives?
    # No, look at `eval_hits` in collab.py:
    # "y_pred_neg is an array. rank y_pred_pos[i] against y_pred_neg for each i"
    # It takes `kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]`
    # This implies there is a GLOBAL set of negative scores.
    # And we check for each positive score if it is greater than the K-th best negative score.
    # This logic seems to calculate Hits@K by checking if the positive score is in the top K relative to the pool of Negatives.
    # Standard OGB Evaluator usually does per-sample ranking (1 pos vs 1000 negs).
    # But `collab.py` manual `eval_hits` seems to treat `y_pred_neg` as a shared pool or aggregated stats?
    # Actually `torch.topk(y_pred_neg, K)` gets the top K scores from ALL negative predictions.
    # Then `y_pred_pos > kth...` checks if the positive score is higher than the K-th highest negative score.
    # This is slightly different from standard Hits@K where you rank 1 Pos among N Negs.
    
    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    return {'hits@{}'.format(K): hitsK}

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def train(model, x, edge_index, edge_weight, train_pos_edge, optimizer, pred, embedding=None, pseudo_pos_dict=None):
    model.train()
    pred.train()

    # train_pos_edge is [num_edges, 2]
    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0
    
    # Masking preparation
    if args.maskinput:
        # We need to reconstruct the graph for each batch if masking
        # train_pos_edge contains ALL training edges.
        # We want to remove the edges in the current batch from the Message Passing graph.
        pass

    sample_time = 0
    for _, batch_indices in enumerate(tqdm.tqdm(dataloader)):
        # Construct xemb
        xemb = x
        if embedding is not None:
            xemb = torch.cat((embedding.weight, x), dim=1) if x is not None else embedding.weight
        
        # Determine msg_edge_index
        if args.maskinput:
            # Mask out the current batch edges from the training graph
            # train_pos_edge has all edges.
            # We want all edges EXCEPT train_pos_edge[batch_indices]
            
            mask = torch.ones(train_pos_edge.size(0), dtype=torch.bool, device=train_pos_edge.device)
            mask[batch_indices] = False
            
            # Message Passing Edges
            msg_edges = train_pos_edge[mask]
            
            # If using validation as input, we should have added them to train_pos_edge or keep them separate?
            # In DGL code `if args.use_valid_as_input`, they fused train and valid edges into `train_edge_index`.
            # We will assume `train_pos_edge` here corresponds to the edges we iterate over for loss.
            # But wait, `train_pos_edge` in DGL code is from `split_edge['train']['edge']`. 
            # The GRAPH (g) was constructed separately.
            # The GRAPH `g` in DGL code *might* contain valid edges if `use_valid_as_input` is set.
            # BUT when iterating `train_pos_edge` (which is just TRAIN edges), we mask THOSE out of `g`.
            
            # In our PyG case, we should pass `train_graph_edge_index` to this function, which represents the full graph structure.
            # And `train_pos_edge` which represents the positive samples.
            # If `train_graph_edge_index` == `train_pos_edge` (transposed), then we can mask directly.
            # If `use_valid_as_input` is True, `train_graph_edge_index` includes valid edges.
            
            # Simplified approach: We will assume `edge_index` passed to train() is the FULL graph (Training (+ Valid)).
            # We need to identify which edges in `edge_index` correspond to `pos_edge` (the batch) and remove them.
            # This is expensive to find by value.
            # A better way (used in DGL code):
            # DGL code maintains `mask` over `train_pos_edge`.
            # And constructs `g_mask` from `train_pos_edge[mask]`.
            # So `g` is effectively recreated from `train_pos_edge` (minus batch) (+ valid edges).
            
            remaining_train = train_pos_edge[mask]
            
            # Helper to create symmetric + self loop (similar to DGL pretreatment)
            # DGL: g_mask = dgl.graph((tei[:, 0], tei[:, 1]), ...) -> add self loop
            # Note: DGL code in `maskinput` block does: 
            #   tei = train_pos_edge[mask]
            #   tei = tei.unique(dim=0)
            #   src, dst = tei.t()
            #   re_tei = stack(dst, src) ... cat ...
            #   g_mask = dgl.graph(...)
            
            # We replicate this:
            curr_train_edges = remaining_train
            if args.use_valid_as_input:
                # We need access to valid edges here if we want to include them.
                # Let's assume we handle this by passing a separate `valid_edge_index` or pre-merging outside?
                # The DGL code logic for `train_edge_index` inside `if args.use_valid_as_input` (lines 303-311)
                # merges train and valid into the initial graph.
                # BUT inside the loop (lines 125-136), it reconstructs `g_mask` using `train_pos_edge[mask]`.
                # Does `g_mask` include valid edges?
                # "g_mask = dgl.graph((tei[:, 0], tei[:, 1]), num_nodes=g.num_nodes())"
                # `tei` comes from `train_pos_edge[mask]`.
                # `train_pos_edge` comes from `split_edge['train']['edge']`.
                # So `g_mask` ONLY contains the remaining TRAIN edges. It LOSSES the valid edges if they were in the original `g`.
                # Wait, looking at DGL code carefully:
                # `if args.maskinput:` -> `tei = train_pos_edge[mask]`. 
                # It does NOT seem to add valid edges back in for `g_mask`. 
                # This suggests when `maskinput` is True, `use_valid_as_input` might be effectively ignored for the dynamic graph construction, OR I am missing where `tei` gets valid edges.
                # Ah, `train_pos_edge` is defined `train_pos_edge = split_edge['train']['edge']` (Line 322).
                # So `g_mask` is purely training edges.
                # This might be a bug or feature of the DGL code. I will stick to replicating the logic: Reconstruct from remaining train edges.
                pass 
                
            # Construct batches edge_index
            # Undirected logic
            edge_index_batch = torch.cat([curr_train_edges, curr_train_edges.flip(1)], dim=0).t()
            edge_index_batch, _ = add_self_loops(edge_index_batch, num_nodes=x.size(0))
            
            # Handle weights if present (TODO: if edge weights are crucial, we need to index them too. DGL code uses torch.ones for g_mask weights line 134)
            edge_weight_batch = None # model expects None or tensor. g_mask one is all ones often.
            
            h = model(edge_index_batch, xemb, edge_weight_batch)
            
        else:
            # Use fixed graph
            h = model(edge_index, xemb, edge_weight)

        pos_edge = train_pos_edge[batch_indices] # [B, 2]
        
        st = time.time()
        # Negative Sampling
        # DGL: neg_train_edge = neg_sampler(g, pos_edge.t()[0]) (This is GlobalUniform).
        # We use PyG negative_sampling.
        # We need roughly 1 negative per positive (args.num_neg)
        neg_edge = negative_sampling(
            edge_index=edge_index, # Used to avoid false negatives? Or just random? 
            # GlobalUniform usually just picks random nodes. DGL GlobalUniform doesn't check for collisions by default unless specified?
            # Actually GlobalUniform in DGL samples uniformly from all nodes.
            # PyG negative_sampling checks for collisions with existing edges if provided.
            num_nodes=x.size(0),
            num_neg_samples=pos_edge.size(0) * args.num_neg,
            method='sparse'
        ).t() # [B, 2]
        
        sample_time += time.time() - st
        
        # Predictions
        pos_score = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
        neg_score = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
        
        if args.loss == 'auc':
            loss = auc_loss(pos_score, neg_score, args.num_neg)
        elif args.loss == 'hauc':
            loss = hinge_auc_loss(pos_score, neg_score, args.num_neg)
        elif args.loss == 'rank':
            loss = log_rank_loss(pos_score, neg_score, args.num_neg)
        elif args.loss == 'pull':
            neg_targets = torch.zeros_like(neg_score)
            if pseudo_pos_dict is not None:
                 # Check pseudo-positives
                 # We need to iterate carefully or use map
                 # neg_edge is [B, 2]
                 neg_edge_cpu = neg_edge.cpu().numpy()
                 # This loop is slow in Python but matches original code structure
                 targets = []
                 for i in range(neg_edge.size(0)):
                     u, v = int(neg_edge_cpu[i, 0]), int(neg_edge_cpu[i, 1])
                     if u > v: u, v = v, u
                     val = pseudo_pos_dict.get((u, v), 0.0)
                     targets.append(val)
                 neg_targets = torch.tensor(targets, device=neg_score.device, dtype=neg_score.dtype)
            
            loss = pull_loss(pos_score, neg_score, neg_targets)
        else:
             loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + \
                    F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score)).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        nn.utils.clip_grad_norm_(pred.parameters(), args.clip_norm)
        if embedding is not None:
             nn.utils.clip_grad_norm_(embedding.parameters(), args.clip_norm)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

@torch.no_grad()
def eval(model, x, edge_index, edge_weight, pos_train_edge, pos_valid_edge, neg_valid_edge, pred, embedding=None):
    model.eval()
    pred.eval()
    
    xemb = x
    if embedding is not None:
        xemb = torch.cat((embedding.weight, x), dim=1) if x is not None else embedding.weight

    h = model(edge_index, xemb, edge_weight)
    
    # Valid Positive
    dataloader = DataLoader(range(pos_valid_edge.size(0)), args.batch_size)
    pos_valid_score = []
    total_val_loss = 0
    
    # Validation Loss Loop (mimicking DGL code)
    for _, batch_idx in enumerate(tqdm.tqdm(dataloader)):
        pos_edge = pos_valid_edge[batch_idx]
        pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
        pos_valid_score.append(pos_pred)
        
        # Sample Negatives for Loss
        neg_valid_edge_loss = negative_sampling(
            edge_index=edge_index,
            num_nodes=x.size(0),
            num_neg_samples=pos_edge.size(0) * args.num_neg
        ).t()
        neg_pred_loss = pred(h[neg_valid_edge_loss[:, 0]], h[neg_valid_edge_loss[:, 1]])
        
        if args.loss == 'auc':
            loss = auc_loss(pos_pred, neg_pred_loss, args.num_neg)
        elif args.loss == 'hauc':
            loss = hinge_auc_loss(pos_pred, neg_pred_loss, args.num_neg)
        elif args.loss == 'rank':
            loss = log_rank_loss(pos_pred, neg_pred_loss, args.num_neg)
        else:
            pos_loss = -F.logsigmoid(pos_pred).mean()
            neg_loss = -F.logsigmoid(-neg_pred_loss).mean()
            loss = pos_loss + neg_loss
        total_val_loss += loss.item()
        
    pos_valid_score = torch.cat(pos_valid_score, dim=0)
    val_loss = total_val_loss / len(dataloader)
    
    # Valid Negative
    dataloader = DataLoader(range(neg_valid_edge.size(0)), args.batch_size)
    neg_valid_score = []
    for _, batch_idx in enumerate(tqdm.tqdm(dataloader)):
        neg_edge = neg_valid_edge[batch_idx]
        neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
        neg_valid_score.append(neg_pred)
    neg_valid_score = torch.cat(neg_valid_score, dim=0)
    
    valid_results = {}
    for k in [20, 50, 100]:
        valid_results[f'hits@{k}'] = eval_hits(pos_valid_score, neg_valid_score, k)[f'hits@{k}']
        
    # Train Scores (for comparing against Valid Negatives)
    dataloader = DataLoader(range(pos_train_edge.size(0)), args.batch_size)
    pos_train_score = []
    for _, batch_idx in enumerate(tqdm.tqdm(dataloader)):
         pos_edge = pos_train_edge[batch_idx]
         pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
         pos_train_score.append(pos_pred)
    pos_train_score = torch.cat(pos_train_score, dim=0)
    
    train_results = {}
    for k in [20, 50, 100]:
        train_results[f'hits@{k}'] = eval_hits(pos_train_score, neg_valid_score, k)[f'hits@{k}']
        
    return valid_results, train_results, val_loss

@torch.no_grad()
def test(model, x, edge_index, edge_weight, pos_test_edge, neg_test_edge, pred, embedding=None):
    model.eval()
    pred.eval()
    
    xemb = x
    if embedding is not None:
        xemb = torch.cat((embedding.weight, x), dim=1) if x is not None else embedding.weight

    h = model(edge_index, xemb, edge_weight)

    dataloader = DataLoader(range(pos_test_edge.size(0)), args.batch_size)
    pos_score = []
    for _, batch_idx in enumerate(tqdm.tqdm(dataloader)):
        pos_edge = pos_test_edge[batch_idx]
        pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
        pos_score.append(pos_pred)
    pos_score = torch.cat(pos_score, dim=0)
    
    dataloader = DataLoader(range(neg_test_edge.size(0)), args.batch_size)
    neg_score = []
    for _, batch_idx in enumerate(tqdm.tqdm(dataloader)):
        neg_edge = neg_test_edge[batch_idx]
        neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
        neg_score.append(neg_pred)
    neg_score = torch.cat(neg_score, dim=0)
    
    results = {}
    for k in [20, 50, 100]:
        results[f'hits@{k}'] = eval_hits(pos_score, neg_score, k)[f'hits@{k}']
    return results

# --- Main Execution ---

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

# Load Dataset
dataset = PygLinkPropPredDataset(name=args.dataset, root='dataset')
data = dataset[0]
split_edge = dataset.get_edge_split()

# Prepare Data (ogbl-collab year filtering)
if args.dataset == 'ogbl-collab':
    # PyG split_edge['train'] has 'edge', 'year', 'weight'
    # 'edge' is [N, 2]
    
    # Filter by year
    selected_year_index = (split_edge['train']['year'] >= 2011).nonzero(as_tuple=True)[0]
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    
    train_edge_index = split_edge['train']['edge'].t() # [2, N]
    train_edge_weight = split_edge['train']['weight']

    if args.use_valid_as_input:
        valid_edge_index = split_edge['valid']['edge'].t()
        valid_edge_weight = split_edge['valid']['weight']
        
        # Merge
        train_edge_index = torch.cat([train_edge_index, valid_edge_index], dim=1)
        train_edge_weight = torch.cat([train_edge_weight, valid_edge_weight], dim=0)

    # To Undirected (for Message Passing)
    # Note: ogbl-collab edges are directed in the raw format? No, usually undirected but stored once.
    # PyG GNNs usually expect symmetric edge_index for undirected graphs.
    # We create symmetric graph + self loops
    # Reuse weight if present
    
    edge_index = train_edge_index
    edge_weight = train_edge_weight
    
    # Symmetrize
    # We can use torch_geometric.utils.to_undirected, but we need to handle weights too if we use them
    # For now, let's just stack
    edge_index_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    if edge_weight is not None:
        edge_weight_sym = torch.cat([edge_weight, edge_weight], dim=0)
    else:
        edge_weight_sym = None
        
    # Add Self Loops
    edge_index_sym, edge_weight_sym = add_self_loops(edge_index_sym, edge_weight_sym, num_nodes=data.num_nodes, fill_value=1.0)
    
    # Assign to data
    data.edge_index = edge_index_sym
    data.edge_weight = edge_weight_sym.to(torch.float32)
    
else:
    # Generic handling
    data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)


# Move to device
data = data.to(device)
train_pos_edge = split_edge['train']['edge'].to(device)
valid_pos_edge = split_edge['valid']['edge'].to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
test_pos_edge = split_edge['test']['edge'].to(device)
test_neg_edge = split_edge['test']['edge_neg'].to(device)

# Embeddings
embedding = None
if args.emb_hidden > 0:
    embedding = torch.nn.Embedding(data.num_nodes, args.emb_hidden).to(device)
    if args.init == 'orthogonal':
        torch.nn.init.orthogonal_(embedding.weight)
    elif args.init == 'ones':
        torch.nn.init.ones_(embedding.weight)
    elif args.init == 'random':
        torch.nn.init.uniform_(embedding.weight)

# Predictor
if args.pred == 'dot':
    pred = DotPredictor().to(device)
elif args.pred == 'mlp':
    pred = Hadamard_MLPPredictor(args.hidden, args.dropout, args.mlp_layers, args.res, args.norm, args.scale).to(device)
else:
    raise NotImplementedError

# Model
input_dim = data.x.shape[1] + args.emb_hidden if embedding is not None else data.x.shape[1]
# data.x needs to be handled if None (ogbl-ddi). But ogbl-collab has features.
if data.x is None:
    # If no features, only embedding
    input_dim = args.emb_hidden

if args.model == 'GCN':
    model = GCN_with_feature(input_dim, args.hidden, args.norm, args.dp4norm, args.prop_step, 
                             args.dropout, args.residual, args.concat_skip, args.relu, args.linear, args.conv).to(device)
elif args.model == 'LightGCN':
    model = LightGCN(input_dim, args.hidden, args.prop_step, args.dropout, args.alpha, args.exp, args.relu).to(device)
else:
    raise NotImplementedError

# Optimizer
params = itertools.chain(model.parameters(), pred.parameters())
if embedding is not None:
    params = itertools.chain(params, embedding.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr)

# PULL Setup
if args.loss == 'pull':
    pull_K = train_pos_edge.size(0)
    # Compute degrees (simple node degree from edge_index)
    # data.edge_index is symmetric and has self loops.
    # Just counting occurrences is approximate degree (with self loops).
    # Better to compute in degree from original simple graph or just use data.edge_index
    deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    top_deg_nodes = torch.topk(deg, 100).indices.cpu().numpy()
    pseudo_pos_dict = {}
    pull_updates = 0

best_val = 0
final_test_result = None
best_epoch = 0
best_model_path = f'{args.run_name}.pt' if args.run_name else 'model_pyg.pt'
losses = []
val_losses = []

print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

tot_time = 0
for epoch in range(1, args.epochs + 1):
    
    # PULL Update Logic
    if args.loss == 'pull' and epoch % args.pull_interval == 0 and pull_updates < 10:
        pull_updates += 1
        print(f"PULL Update {pull_updates}: Generating pseudo-labels...")
        
        pull_K = int(pull_K + args.pull_r * train_pos_edge.size(0))
        num_candidates = 5 * pull_K
        
        model.eval()
        with torch.no_grad():
            xemb = data.x
            if embedding is not None:
                xemb = torch.cat((embedding.weight, data.x), dim=1) if data.x is not None else embedding.weight
            
            h = model(data.edge_index, xemb, data.edge_weight)
            
            candidates_src = np.random.choice(top_deg_nodes, num_candidates)
            candidates_dst = np.random.randint(0, data.num_nodes, num_candidates)
            
            # Check existence (Slow on CPU loops? Using mask optimization if possible)
            # We want to check if (src, dst) exists in current graph.
            # PyG utils doesn't have fast 'has_edge' for batch.
            # But we can assume most random pairs are non-edges.
            # Using a set for lookup is faster.
            # Only do this once.
            if 'edge_set' not in locals():
                # Convert edge_index to set of tuples
                # CPU based set
                ei_cpu = data.edge_index.cpu().numpy()
                edge_set = set(zip(ei_cpu[0], ei_cpu[1]))
            
            valid_src_list = []
            valid_dst_list = []
            for u, v in zip(candidates_src, candidates_dst):
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    valid_src_list.append(u)
                    valid_dst_list.append(v)
            
            valid_src = torch.tensor(valid_src_list, device=device)
            valid_dst = torch.tensor(valid_dst_list, device=device)
            
            if len(valid_src) == 0:
                print("Warning: No valid candidates found for PULL.")
                pseudo_pos_dict = {}
            else:
                 cand_h_src = h[valid_src]
                 cand_h_dst = h[valid_dst]
                 cand_scores = pred(cand_h_src, cand_h_dst).sigmoid().cpu().numpy()
                 
                 k_actual = min(len(cand_scores), pull_K)
                 top_indices = np.argpartition(cand_scores, -k_actual)[-k_actual:]
                 
                 pseudo_pos_dict = {}
                 for idx in top_indices:
                     u, v = int(valid_src[idx]), int(valid_dst[idx])
                     score = float(cand_scores[idx])
                     if u > v: u, v = v, u
                     pseudo_pos_dict[(u, v)] = score
                 print(f"PULL Update: Selected {len(pseudo_pos_dict)} pseudo-positives")

    st = time.time()
    
    loss_args = [model, data.x, data.edge_index, data.edge_weight, train_pos_edge, optimizer, pred, embedding]
    if args.loss == 'pull':
        loss_args.append(pseudo_pos_dict)
        
    loss = train(*loss_args)
    
    epoch_time = time.time() - st
    tot_time += epoch_time
    losses.append(loss)
    
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
        
    valid_results, train_results, val_loss = eval(model, data.x, data.edge_index, data.edge_weight, 
                                                  train_pos_edge, valid_pos_edge, valid_neg_edge, pred, embedding)
    
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train {args.metric}: {train_results[args.metric]:.4f}, Valid {args.metric}: {valid_results[args.metric]:.4f}")
    
    # Checkpoint
    if valid_results[args.metric] >= best_val:
        best_val = valid_results[args.metric]
        best_epoch = epoch
        
        # Test
        # For test, we might want to include valid edges in the graph if not already included?
        # If use_valid_as_input is True, they are already in data.edge_index.
        # If False, they are NOT. We should add them for testing.
        if not args.use_valid_as_input:
             # Construct test graph = Data + Valid Edges
             # We create a temporary edge_index
             val_ei = split_edge['valid']['edge'].t().to(device)
             test_edge_index = torch.cat([data.edge_index, val_ei, val_ei.flip(0)], dim=1)
             test_edge_weight = None
             if data.edge_weight is not None:
                 w_val = torch.ones(val_ei.size(1) * 2, device=device) # Assume weight 1
                 test_edge_weight = torch.cat([data.edge_weight, w_val], dim=0)
        else:
             test_edge_index = data.edge_index
             test_edge_weight = data.edge_weight
             
        final_test_result = test(model, data.x, test_edge_index, test_edge_weight, test_pos_edge, test_neg_edge, pred, embedding)
        
        torch.save({'model': model.state_dict(), 'pred': pred.state_dict()}, best_model_path)
        
    wandb.log({'loss': loss, 'val_loss': val_loss, 'train_hit': train_results[args.metric], 'valid_hit': valid_results[args.metric]})
    
    if epoch - best_epoch >= 200:
        break

print(f"Final Test {args.metric}: {final_test_result[args.metric]:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.savefig('learning_curve_pyg.png')
wandb.log({"learning_curve": wandb.Image('learning_curve_pyg.png')})

if args.hf_repo_id:
    from huggingface_hub import HfApi
    api = HfApi(token=args.hf_token)
    api.upload_file(
        path_or_fileobj=best_model_path,
        path_in_repo=best_model_path,
        repo_id=args.hf_repo_id,
        repo_type="model"
    )
