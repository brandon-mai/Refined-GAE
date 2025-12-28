import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.utils.data import DataLoader
import tqdm
import argparse
from loss import auc_loss, hinge_auc_loss, log_rank_loss, pull_loss
from model import GCN_with_feature, DotPredictor, LightGCN, Hadamard_MLPPredictor, GCN_with_feature_multilayers
import time
import wandb
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, upload_file
import numpy as np


def parse():
    
    parser = argparse.ArgumentParser()
    # General Setup
    parser.add_argument("--dataset", default='ogbl-collab', choices=['ogbl-collab'], type=str, help="Dataset name (used in Main Script -> Data Loading)")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device ID (used in Main Script -> Device Selection)")
    parser.add_argument("--seed", default=0, type=int, help="Random seed (used in Main Script -> Random Seed Setup)")
    parser.add_argument('--run_name', default=None, type=str, help="WandB run name (used in Main Script -> WandB Init)")
    parser.add_argument('--hf_token', default=None, type=str, help="Hugging Face token (used in Main Script -> Hugging Face Hub)")
    parser.add_argument('--hf_repo_id', default=None, type=str, help="Hugging Face repository ID (used in Main Script -> Hugging Face Hub)")

    # Training Configuration
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate (used in Main Script -> Optimizer)")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs (used in Main Script -> Training Loop)")
    parser.add_argument("--batch_size", default=8192, type=int, help="Batch size (used in Main Script -> DataLoader)")
    parser.add_argument("--metric", default='hits@20', type=str, help="Evaluation metric (used in Main Script -> Evaluation)")
    parser.add_argument("--loss", default='bce', choices=['bce', 'auc', 'hauc', 'rank', 'pull'], type=str, help="Loss function type (used in Main Script -> Loss Calculation)")
    parser.add_argument("--interval", default=100, type=int, help="Interval for learning rate decay (used in Main Script -> Training Loop)")
    parser.add_argument("--step_lr_decay", action='store_true', default=True, help="Whether to use step learning rate decay (used in Main Script -> Learning Rate Scheduler)")
    parser.add_argument('--clip_norm', default=1.0, type=float, help="Gradient clipping norm (used in Main Script -> Training Loop)")

    # Data / Sampling
    parser.add_argument("--num_neg", default=1, type=int, help="Number of negative samples per positive sample (used in train loop)")
    parser.add_argument("--maskinput", action='store_true', help="Whether to use input masking (used in train loop)")
    parser.add_argument("--use_valid_as_input", action='store_true', help="Whether to use validation set as input (used in Main Script -> Data Loading)")
    parser.add_argument("--pull_interval", default=200, type=int, help="Interval for PULL expected graph update (used in train loop)")
    parser.add_argument("--pull_r", default=0.05, type=float, help="Growth rate for PULL K (used in train loop)")

    # Encoder (GCN / Embedding)
    parser.add_argument("--model", default='GCN', type=str, help="Model architecture type (used in Main Script -> Model Instantiation)")
    parser.add_argument("--conv", default='GCN', type=str, help="Graph convolution type (used in model.py -> GCN)")
    parser.add_argument("--emb_hidden", default=64, type=int, help="Hidden dimension for embedding (used in Main Script -> Node Embeddings)")
    parser.add_argument("--hidden", default=64, type=int, help="Hidden dimension size (used in model.py -> GCN, MLPPredictor)")
    parser.add_argument("--prop_step", default=8, type=int, help="Number of propagation steps in GCN (used in model.py -> GCN, LightGCN)")
    parser.add_argument("--dropout", default=0.05, type=float, help="Dropout rate (used in model.py -> GCN, MLPPredictor)")
    parser.add_argument("--norm", action='store_true', default=False, help="Whether to use normalization (used in model.py -> GCN)")
    parser.add_argument("--dp4norm", default=0, type=float, help="Dropout for normalization (Note: Passed to model.py but seemingly unused in GCN_with_feature)")
    parser.add_argument("--dpe", default=0, type=float, help="Edge dropout rate (Unused in this script)")
    parser.add_argument("--drop_edge", action='store_true', default=False, help="Whether to enable edge dropout (Unused in this script)")
    parser.add_argument("--residual", default=0, type=float, help="Residual connection weight (used in model.py -> GCN)")
    parser.add_argument('--linear', action='store_true', default=False, help="Whether to use linear layers in GCN (used in model.py -> GCN)")
    parser.add_argument('--alpha', default=0.5, type=float, help="Alpha for LightGCN (used in model.py -> LightGCN)")
    parser.add_argument('--exp', action='store_true', default=False, help="Exponential decay for LightGCN (used in model.py -> LightGCN)")
    parser.add_argument('--init', default='orthogonal', type=str, help="Initialization method (used in Main Script -> Embedding Init)")
    parser.add_argument("--relu", action='store_true', default=False, help="Whether to use ReLU (used in model.py -> GCN, LightGCN)")
    parser.add_argument('--concat_skip', action='store_true', default=False, help='use skip-concat in encoder instead of residual/add')

    # Decoder (Predictor)
    parser.add_argument("--pred", default='mlp', type=str, help="Predictor type (used in Main Script -> Model Instantiation)")
    parser.add_argument("--mlp_layers", default=2, type=int, help="Number of MLP layers (used in model.py -> MLPPredictor)")
    parser.add_argument("--res", action='store_true', default=False, help="Whether to use residual connections in predictor (used in model.py -> MLPPredictor)")
    parser.add_argument('--scale', action='store_true', default=False, help="Whether to scale inputs in predictor (used in model.py -> MLPPredictor)")


    args = parser.parse_args()
    return args

args = parse()
print(args)

if args.run_name:
    wandb.init(project='Refined-GAE', name=args.run_name, config=args)
else:
    wandb.init(project='Refined-GAE', config=args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dgl.seed(args.seed)

def eval_hits(y_pred_pos, y_pred_neg, K):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''

    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    return {'hits@{}'.format(K): hitsK}

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def train(model, g, train_pos_edge, optimizer, neg_sampler, pred, scaler, embedding=None, pseudo_pos_dict=None):
    model.train()
    pred.train()

    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0
    if args.maskinput:
        mask = torch.ones(train_pos_edge.size(0), dtype=torch.bool)
    sample_time = 0
    for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
        xemb = torch.cat((embedding.weight, g.ndata['feat']), dim=1) if embedding is not None else g.ndata['feat']
        # Use AMP
        with torch.cuda.amp.autocast(enabled=True):
            if args.maskinput:
                mask[edge_index] = 0
                tei = train_pos_edge[mask]
                tei = tei.unique(dim=0)
                src, dst = tei.t()
                re_tei = torch.stack((dst, src), dim=0).t()
                tei = torch.cat((tei, re_tei), dim=0)
                g_mask = dgl.graph((tei[:, 0], tei[:, 1]), num_nodes=g.num_nodes())
                g_mask = dgl.add_self_loop(g_mask)
                edge_weight = torch.ones(g_mask.number_of_edges(), dtype=torch.float32).to(device)
                h = model(g_mask, xemb, edge_weight)
                mask[edge_index] = 1
            else:
                h = model(g, xemb, g.edata['weight'])

            pos_edge = train_pos_edge[edge_index]
            neg_train_edge = neg_sampler(g, pos_edge.t()[0])
            neg_train_edge = torch.stack(neg_train_edge, dim=0)
            neg_train_edge = neg_train_edge.t()
            neg_edge = neg_train_edge
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
                    neg_edge_cpu = neg_edge.cpu()
                    for i in range(neg_edge.size(0)):
                        u, v = int(neg_edge_cpu[i, 0]), int(neg_edge_cpu[i, 1])
                        if u > v: u, v = v, u 
                        if (u, v) in pseudo_pos_dict:
                            neg_targets[i] = pseudo_pos_dict[(u, v)]
                loss = pull_loss(pos_score, neg_score, neg_targets)
            else:
            # pos_loss = -F.logsigmoid(pos_score).mean()
            # neg_loss = -F.logsigmoid(-neg_score).mean()
            # loss = pos_loss + neg_loss
                loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score)).mean()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        torch.nn.utils.clip_grad_norm_(pred.parameters(), args.clip_norm)
        if embedding is not None:
            torch.nn.utils.clip_grad_norm_(embedding.parameters(), args.clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        # print(f"Sample time: {sample_time:.4f}", flush=True)

    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, pred, embedding=None):
    model.eval()
    pred.eval()

    with torch.no_grad():
        xemb = torch.cat((embedding.weight, g.ndata['feat']), dim=1) if embedding is not None else g.ndata['feat']
        h = model(g, xemb, g.edata['weight'])
        dataloader = DataLoader(range(pos_test_edge.size(0)), args.batch_size)
        pos_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_test_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score.append(pos_pred)
        pos_score = torch.cat(pos_score, dim=0)
        dataloader = DataLoader(range(neg_test_edge.size(0)), args.batch_size)
        neg_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            neg_edge = neg_test_edge[edge_index]
            neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
            neg_score.append(neg_pred)
        neg_score = torch.cat(neg_score, dim=0)
        results = {}
        for k in [20, 50, 100]:
            results[f'hits@{k}'] = eval_hits(pos_score, neg_score, k)[f'hits@{k}']
    return results

def eval(model, g, pos_train_edge, pos_valid_edge, neg_valid_edge, pred, embedding=None):
    model.eval()
    pred.eval()
    
    # Negative sampler for validation loss calculation (mimicking train)
    neg_sampler = GlobalUniform(args.num_neg)

    with torch.no_grad():
        xemb = torch.cat((embedding.weight, g.ndata['feat']), dim=1) if embedding is not None else g.ndata['feat']
        h = model(g, xemb, g.edata['weight'])
        
        # Validation Positive Scores & Loss Calculation
        dataloader = DataLoader(range(pos_valid_edge.size(0)), args.batch_size)
        pos_valid_score = []
        total_val_loss = 0
        
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_valid_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_valid_score.append(pos_pred)
            
            # Validation Loss Calculation (mimicking train loop)
            # Sample negatives for this batch of positives
            neg_valid_edge_loss = neg_sampler(g, pos_edge.t()[0])
            neg_valid_edge_loss = torch.stack(neg_valid_edge_loss, dim=0).t()
            
            neg_pred_loss = pred(h[neg_valid_edge_loss[:, 0]], h[neg_valid_edge_loss[:, 1]])
            
            if args.loss == 'auc':
                loss = auc_loss(pos_pred, neg_pred_loss, args.num_neg)
            elif args.loss == 'hauc':
                loss = hinge_auc_loss(pos_pred, neg_pred_loss, args.num_neg)
            elif args.loss == 'rank':
                loss = log_rank_loss(pos_pred, neg_pred_loss, args.num_neg)
            else: # pull loss not typically used for validation metrics directly
                pos_loss = -F.logsigmoid(pos_pred).mean()
                neg_loss = -F.logsigmoid(-neg_pred_loss).mean()
                loss = pos_loss + neg_loss
            
            total_val_loss += loss.item()

        pos_valid_score = torch.cat(pos_valid_score, dim=0)
        val_loss = total_val_loss / len(dataloader)
        
        # Validation Negative Scores (for Metrics)
        dataloader = DataLoader(range(neg_valid_edge.size(0)), args.batch_size)
        neg_valid_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            neg_edge = neg_valid_edge[edge_index]
            neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
            neg_valid_score.append(neg_pred)
        neg_valid_score = torch.cat(neg_valid_score, dim=0)
        
        valid_results = {}
        for k in [20, 50, 100]:
            valid_results[f'hits@{k}'] = eval_hits(pos_valid_score, neg_valid_score, k)[f'hits@{k}']
        
        dataloader = DataLoader(range(pos_train_edge.size(0)), args.batch_size)
        pos_train_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_train_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_train_score.append(pos_pred)
        pos_train_score = torch.cat(pos_train_score, dim=0)
        
        train_results = {}
        for k in [20, 50, 100]:
            train_results[f'hits@{k}'] = eval_hits(pos_train_score, neg_valid_score, k)[f'hits@{k}']
            
    return valid_results, train_results, val_loss

# Load the dataset
dataset = DglLinkPropPredDataset(name=args.dataset)
split_edge = dataset.get_edge_split()

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

graph = dataset[0]

if args.dataset == 'ogbl-collab':
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= 2011).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()

    if args.use_valid_as_input:
        u, v = split_edge['valid']['edge'].t()
        re_vei = torch.stack((v, u), dim=0)

        u, v = split_edge['train']['edge'].t()
        re_tei = torch.stack((v, u), dim=0)

        train_edge_index = torch.cat((train_edge_index, re_tei, re_vei, split_edge['valid']['edge'].t()), dim=1)
        train_edge_weight = torch.cat((split_edge['train']['weight'], split_edge['train']['weight'], split_edge['valid']['weight'], split_edge['valid']['weight']), dim=0)

        feat = graph.ndata['feat']
        graph = dgl.graph((train_edge_index[0], train_edge_index[1]), num_nodes=graph.num_nodes())
        graph.ndata['feat'] = feat
        graph.edata['weight'] = train_edge_weight.to(torch.float32)
        split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], split_edge['valid']['edge']), dim=0)
        graph = dgl.add_self_loop(graph, fill_data=0)
        graph = graph.to(device)
    else:
        feat = graph.ndata['feat']
        graph = dgl.graph((train_edge_index[0], train_edge_index[1]), num_nodes=graph.num_nodes())
        graph.ndata['feat'] = feat
        graph = dgl.add_self_loop(graph)
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph.edata['weight'] = torch.ones(graph.number_of_edges(), dtype=torch.float32)
        graph = graph.to(device)

train_pos_edge = split_edge['train']['edge'].to(device)
valid_pos_edge = split_edge['valid']['edge'].to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
test_pos_edge = split_edge['test']['edge'].to(device)
test_neg_edge = split_edge['test']['edge_neg'].to(device)

if args.emb_hidden > 0:
    embedding = torch.nn.Embedding(graph.num_nodes(), args.emb_hidden).to(device)
    if args.init == 'orthogonal':
        torch.nn.init.orthogonal_(embedding.weight)
    elif args.init == 'ones':
        torch.nn.init.ones_(embedding.weight)
    elif args.init == 'random':
        torch.nn.init.uniform_(embedding.weight)
else:
    embedding = None

# Create negative samples for training
neg_sampler = GlobalUniform(args.num_neg)

if args.pred == 'dot':
    pred = DotPredictor().to(device)
elif args.pred == 'mlp':
    pred = Hadamard_MLPPredictor(args.hidden, args.dropout, args.mlp_layers, args.res, args.norm, args.scale).to(device)
else:
    raise NotImplementedError

input_dim = graph.ndata['feat'].shape[1] + args.emb_hidden if embedding is not None else graph.ndata['feat'].shape[1]

if args.model == 'GCN':
    model = GCN_with_feature(input_dim, args.hidden, args.norm, args.dropout, args.prop_step, args.dropout, args.residual, args.concat_skip, args.relu, args.linear, args.conv).to(device)
elif args.model == 'LightGCN':
    model = LightGCN(input_dim, args.hidden, args.prop_step, args.dropout, args.alpha, args.exp, args.relu).to(device)
elif args.model == 'GCN_multilayer':
    model = GCN_with_feature_multilayers(input_dim, args.hidden, args.norm, args.dropout, args.prop_step, args.dropout, args.residual, args.relu, args.linear).to(device)
else:
    raise NotImplementedError


parameter = itertools.chain(model.parameters(), pred.parameters())
if embedding is not None:
    parameter = itertools.chain(parameter, embedding.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)
scaler = torch.cuda.amp.GradScaler()

# PULL Setup
if args.loss == 'pull':
    pull_K = train_pos_edge.size(0) 
    degs = graph.in_degrees()
    top_deg_nodes = torch.topk(degs, 100).indices.cpu().numpy()
    pseudo_pos_dict = {}
    pull_updates = 0

evaluator = Evaluator(name=args.dataset)

best_val = 0
final_test_result = None
best_epoch = 0

if args.run_name:
    best_model_path = f'{args.run_name}.pt'
else:
    best_model_path = 'model.pt'


losses = []
valid_list = []
test_list = []

if embedding is not None:
    print(f'number of parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in pred.parameters()) + sum(p.numel() for p in embedding.parameters())}')
else:
    print(f'number of parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in pred.parameters())}')

tot_time = 0
for epoch in range(1, args.epochs + 1):
    
    # PULL Outer Loop (Update Pseudo-Labels)
    if args.loss == 'pull' and epoch % args.pull_interval == 0 and pull_updates < 10:
        pull_updates += 1
        print(f"PULL Update {pull_updates}: Generating pseudo-labels...")
        
        pull_K = int(pull_K + args.pull_r * train_pos_edge.size(0))
        num_candidates = 5 * pull_K 
        
        model.eval()
        with torch.no_grad():
            xemb = torch.cat((embedding.weight, graph.ndata['feat']), dim=1) if embedding is not None else graph.ndata['feat']
            h = model(graph, xemb, graph.edata['weight'])
 
            candidates_src = np.random.choice(top_deg_nodes, num_candidates)
            candidates_dst = np.random.randint(0, graph.num_nodes(), num_candidates)
            
            cand_edges = (torch.tensor(candidates_src).to(device), torch.tensor(candidates_dst).to(device))
            exists = graph.has_edges_between(cand_edges[0], cand_edges[1])
            exists_rev = graph.has_edges_between(cand_edges[1], cand_edges[0])
            valid_mask = ~(exists | exists_rev)
            
            valid_src = candidates_src[valid_mask.cpu().numpy()]
            valid_dst = candidates_dst[valid_mask.cpu().numpy()]
            
            if len(valid_src) == 0:
                print("Warning: No valid candidates found for PULL.")
                pseudo_pos_dict = {}
            else:
                # Batched prediction to avoid OOM
                cand_scores_list = []
                batch_size_pull = 10000 
                with torch.cuda.amp.autocast(enabled=True):
                    for i in range(0, len(valid_src), batch_size_pull):
                        batch_src = valid_src[i:i+batch_size_pull]
                        batch_dst = valid_dst[i:i+batch_size_pull]
                        batch_h_src = h[batch_src]
                        batch_h_dst = h[batch_dst]
                        batch_scores = pred(batch_h_src, batch_h_dst).sigmoid().flatten()
                        cand_scores_list.append(batch_scores)
                
                cand_scores = torch.cat(cand_scores_list).cpu().numpy()
                
                k_actual = min(len(cand_scores), pull_K)
                top_indices = np.argpartition(cand_scores, -k_actual)[-k_actual:]
                
                pseudo_pos_dict = {}
                for idx in top_indices:
                    u, v = int(valid_src[idx]), int(valid_dst[idx])
                    score = float(cand_scores[idx])
                    if u > v: u, v = v, u 
                    pseudo_pos_dict[(u, v)] = score
                
                print(f"PULL Update: Selected {len(pseudo_pos_dict)} pseudo-positives (Target K={pull_K})")

    st = time.time()
    if args.loss == 'pull':
        loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred, scaler, embedding, pseudo_pos_dict)
    else:
        loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred, scaler, embedding)
    print(f"Epoch {epoch}, Time: {time.time()-st:.4f}", flush=True)
    tot_time += time.time() - st
    losses.append(loss)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
    valid_results, train_results, val_loss = eval(model, graph, train_pos_edge, valid_pos_edge, valid_neg_edge, pred, embedding)
    valid_list.append(valid_results[args.metric])
    
    # Store validation loss
    if 'val_losses' not in locals():
        val_losses = []
    val_losses.append(val_loss)
    for k, v in valid_results.items():
        print(f'Validation {k}: {v:.4f}')
    for k, v in train_results.items():
        print(f'Train {k}: {v:.4f}')
    if not args.use_valid_as_input:
        graph_t = graph.clone()
        u, v = valid_pos_edge.t()
        graph_t.add_edges(u, v)
        graph_t.add_edges(v, u)
        graph_t.edata['weight'] = torch.ones(graph_t.number_of_edges(), dtype=torch.float32, device=device)
    else:
        graph_t = graph
    test_results = test(model, graph_t, test_pos_edge, test_neg_edge, pred, embedding)
    test_list.append(test_results[args.metric])
    for k, v in test_results.items():
        print(f'Test {k}: {v:.4f}')

    if valid_results[args.metric] >= best_val:
        best_val = valid_results[args.metric]
        best_epoch = epoch
        final_test_result = test_results
        torch.save({'model': model.state_dict(), 'pred': pred.state_dict(), 'embedding': embedding.state_dict() if embedding is not None else None}, best_model_path)

    if epoch - best_epoch >= 200:
        break
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train {args.metric}: {train_results[args.metric]:.4f}, Valid {args.metric}: {valid_results[args.metric]:.4f}, Test {args.metric}: {test_results[args.metric]:.4f}")
    wandb.log({'loss': loss, 'val_loss': val_loss, 'train_hit': train_results[args.metric], 'valid_hit': valid_results[args.metric], 'test_hit': test_results[args.metric]})

print(f'total time: {tot_time:.4f}')
print(f"Test hit: {final_test_result[args.metric]:.4f}")
wandb.log({'final_test_hit': final_test_result[args.metric]})

# Plot Train vs Val Loss
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve.png')
wandb.log({"learning_curve": wandb.Image('learning_curve.png')})

if args.hf_repo_id:
    print(f"Uploading model to Hugging Face: {args.hf_repo_id}")
    api = HfApi(token=args.hf_token)
    api.create_repo(repo_id=args.hf_repo_id, exist_ok=True, repo_type="model")
    api.upload_file(
        path_or_fileobj=best_model_path,
        path_in_repo=best_model_path,
        repo_id=args.hf_repo_id,
        repo_type="model"
    )
    # Also upload the learning curve
    api.upload_file(
        path_or_fileobj='learning_curve.png',
        path_in_repo=f'learning_curve_{args.run_name}.png' if args.run_name else 'learning_curve.png',
        repo_id=args.hf_repo_id,
        repo_type="model"
    )
    print("Upload complete!")


