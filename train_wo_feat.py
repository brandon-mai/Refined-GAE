import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.utils.data import DataLoader
import tqdm
import argparse
from loss import auc_loss, hinge_auc_loss, log_rank_loss, pull_loss
from model import Hadamard_MLPPredictor, GCN, GCN_v1, DotPredictor, LorentzPredictor
import wandb
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, upload_file
import numpy as np


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

def parse():
    
    parser = argparse.ArgumentParser()
    # General Setup
    parser.add_argument("--dataset", default='ogbl-ddi', choices=['ogbl-ddi', 'ogbl-ppa', 'ogbl-collab'], type=str, help="Dataset name (used in Main Script -> Data Loading)")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device ID (used in Main Script -> Device Selection)")
    parser.add_argument("--seed", default=0, type=int, help="Random seed (used in Main Script -> Random Seed Setup)")
    parser.add_argument('--run_name', default=None, type=str, help="WandB run name (used in Main Script -> WandB Init)")
    parser.add_argument('--hf_token', default=None, type=str, help="Hugging Face token (used in Main Script -> Hugging Face Hub)")
    parser.add_argument('--hf_repo_id', default=None, type=str, help="Hugging Face repository ID (used in Main Script -> Hugging Face Hub)")

    # Training Configuration
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate (used in Main Script -> Optimizer)")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs (used in Main Script -> Training Loop)")
    parser.add_argument("--batch_size", default=8192, type=int, help="Batch size for training and evaluation (used in Main Script -> DataLoader)")
    parser.add_argument("--metric", default='hits@20', type=str, help="Evaluation metric (used in Main Script -> Evaluation)")
    parser.add_argument("--loss", default='bce', choices=['bce', 'auc', 'hauc', 'rank', 'pull'], type=str, help="Loss function type (used in Main Script -> Loss Calculation)")
    parser.add_argument("--interval", default=100, type=int, help="Interval for learning rate decay (used in Main Script -> Training Loop)")
    parser.add_argument("--step_lr_decay", action='store_true', default=True, help="Whether to use step learning rate decay (used in Main Script -> Learning Rate Scheduler)")
    parser.add_argument('--clip_norm', default=1.0, type=float, help="Gradient clipping norm (used in Main Script -> Training Loop)")

    # Data / Sampling
    parser.add_argument("--num_neg", default=1, type=int, help="Number of negative samples per positive sample (used in Main Script -> Data Loading / Negative Sampling)")
    parser.add_argument("--maskinput", action='store_true', default=False, help="Whether to mask input edges during training (used in Main Script -> Data Preprocessing)")
    parser.add_argument("--pull_interval", default=200, type=int, help="Interval for PULL expected graph update (used in train loop)")
    parser.add_argument("--pull_r", default=0.05, type=float, help="Growth rate for PULL K (used in train loop)")

    # Encoder (GCN / Embedding)
    parser.add_argument("--model", default='GCN', choices=['GCN', 'GCN_with_feature', 'GCN_with_MLP', 'GCN_v1', 'LightGCN', 'LightGCN_res', 'MultiheadLightGCN'], type=str, help="Model architecture (used in Main Script -> Model Instantiation)")
    parser.add_argument('--conv', default='GCN', choices=['GCN', 'GAT', 'GIN', 'SAGE'], type=str, help="Graph convolution type (used in model.py -> GCN)")
    parser.add_argument("--emb_dim", default=32, type=int, help="Embedding dimension (used in Main Script -> Node Embeddings)")
    parser.add_argument("--hidden", default=32, type=int, help="Hidden dimension size (used in model.py -> GCN, MLPPredictor)")
    parser.add_argument("--prop_step", default=8, type=int, help="Number of propagation steps in GCN (used in model.py -> GCN, LightGCN)")
    parser.add_argument("--dropout", default=0.05, type=float, help="Dropout rate (used in model.py -> GCN, MLPPredictor)")
    parser.add_argument("--norm", action='store_true', default=False, help="Whether to use normalization in GCN (used in model.py -> GCN)")
    parser.add_argument("--dp4norm", default=0, type=float, help="Dropout rate after normalization (used in model.py -> GCN)")
    parser.add_argument("--dpe", default=0, type=float, help="Edge dropout rate (Note: seemingly unused in this script's model init)")
    parser.add_argument("--drop_edge", action='store_true', default=False, help="Whether to enable edge dropout (used in model.py -> GCN)")
    parser.add_argument("--residual", default=0, type=float, help="Residual connection weight (used in model.py -> GCN)")
    parser.add_argument('--linear', action='store_true', default=False, help="Whether to use linear layers in GCN (used in model.py -> GCN)")
    parser.add_argument('--alpha', default=0.5, type=float, help="Alpha parameter for LightGCN (used in model.py -> LightGCN)")
    parser.add_argument('--exp', action='store_true', default=False, help="Exponential decay weighting for LightGCN (used in model.py -> LightGCN)")
    parser.add_argument('--gin_aggr', default='sum', choices=['mean', 'sum'], type=str, help="Aggregation type for GIN (used in model.py -> GCN)")
    parser.add_argument('--multilayer', action='store_true', default=False, help="Whether to use multilayer GCN (used in model.py -> GCN)")
    parser.add_argument('--res', action='store_true', default=False, help="Whether to use residual connections in GCN_v1 (used in model.py -> GCN_v1)")
    parser.add_argument('--init', default='orthogonal', choices=['orthogonal', 'uniform', 'ones'], type=str, help="Embedding initialization method (used in Main Script -> Embedding Init)")
    parser.add_argument('--force_orthogonal', action='store_true', default=False, help="Force orthogonal constraint on embeddings (used in Main Script -> Loss Calculation)")
    parser.add_argument("--relu", action='store_true', default=False, help="Whether to use ReLU activation in GCN (used in model.py -> GCN, LightGCN)")
    parser.add_argument('--concat_skip', action='store_true', default=False, help='use skip-concat in encoder instead of residual/add')

    # Decoder (Predictor)
    parser.add_argument('--pred', default='Hadamard', choices=['Hadamard', 'Dot', 'Lorentz', 'AttMLP', 'Block'], type=str, help="Predictor type (used in Main Script -> Model Instantiation)")
    parser.add_argument('--mlp_layers', type=int, default=2, help="Number of layers in MLP predictor (used in model.py -> MLPPredictor)")
    parser.add_argument('--mlp_res', action='store_true', default=False, help="Whether to use residual connections in MLP predictor (used in model.py -> MLPPredictor)")
    parser.add_argument("--mlp_norm", action='store_true', default=False, help="Whether to use normalization in MLP predictor (used in model.py -> MLPPredictor)")


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

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def train(model, g, train_pos_edge, optimizer, neg_sampler, pred, pseudo_pos_dict=None):
    model.train()
    pred.train()

    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0
    if args.maskinput:
        mask = torch.ones(train_pos_edge.size(0), dtype=torch.bool)
    for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
        if args.maskinput:
            mask[edge_index] = 0
            tei = train_pos_edge[mask]
            src, dst = tei.t()
            re_tei = torch.stack((dst, src), dim=0).t()
            tei = torch.cat((tei, re_tei), dim=0)
            g_mask = dgl.graph((tei[:, 0], tei[:, 1]), num_nodes=g.num_nodes())
            g_mask = dgl.add_self_loop(g_mask)
            h = model(g_mask, g.ndata['feat'])
            mask[edge_index] = 1
        else:
            h = model(g, g.ndata['feat'])

        pos_edge = train_pos_edge[edge_index]
        neg_train_edge = neg_sampler(g, pos_edge.t()[0])
        neg_train_edge = torch.stack(neg_train_edge, dim=0)
        neg_train_edge = neg_train_edge.t()
        neg_edge = neg_train_edge

        pos_score = pred(h[pos_edge[:,0]], h[pos_edge[:,1]])
        neg_score = pred(h[neg_edge[:,0]], h[neg_edge[:,1]])
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
                    if u > v: u, v = v, u # normalize for undirected lookup if we stored them sorted
                    if (u, v) in pseudo_pos_dict:
                        neg_targets[i] = pseudo_pos_dict[(u, v)]
            
            loss = pull_loss(pos_score, neg_score, neg_targets)
        else:
            loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        if args.force_orthogonal:
            loss += 1e-8 * torch.norm(h @ h.t() - torch.diag(torch.diag(h @ h.t())), p='fro')
        
        optimizer.zero_grad()
        loss.backward()
        if args.dataset == 'ogbl-ddi' or args.dataset == 'ogbl-collab':
            torch.nn.utils.clip_grad_norm_(g.ndata['feat'], args.clip_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            torch.nn.utils.clip_grad_norm_(pred.parameters(), args.clip_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, pred):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, g.ndata['feat'])
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

def eval(model, g, pos_train_edge, pos_valid_edge, neg_valid_edge, pred):
    model.eval()
    pred.eval()
    
    # Negative sampler for validation loss calculation (mimicking train)
    neg_sampler = GlobalUniform(args.num_neg)

    with torch.no_grad():
        h = model(g, g.ndata['feat'])
        
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
            
            if args.force_orthogonal:
                 loss += 1e-8 * torch.norm(h @ h.t() - torch.diag(torch.diag(h @ h.t())), p='fro')

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
            
        # Train Positive Scores
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

def plot_dot_product_dist(x):
    dot_products = x @ x.t()
    dot_products = dot_products.detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.hist(dot_products.flatten(), bins=100)
    plt.xlabel('Dot Product')
    plt.ylabel('Frequency')
    plt.title('Dot Product Distribution')
    plt.show()
    plt.savefig('dot_product_distribution.png')

# Load the dataset 
dataset = DglLinkPropPredDataset(name=args.dataset)
split_edge = dataset.get_edge_split()

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

graph = dataset[0]
graph = dgl.add_self_loop(graph).to(device)
#graph = dgl.to_bidirected(graph, copy_ndata=True).to(device)

train_pos_edge = split_edge['train']['edge'].to(device)
valid_pos_edge = split_edge['valid']['edge'].to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
test_pos_edge = split_edge['test']['edge'].to(device)
test_neg_edge = split_edge['test']['edge_neg'].to(device)


# Create negative samples for training
neg_sampler = GlobalUniform(args.num_neg)

if args.pred == 'Hadamard':
    pred = Hadamard_MLPPredictor(args.hidden, args.dropout, args.mlp_layers, args.mlp_res, args.mlp_norm).to(device)
elif args.pred == 'Dot':
    pred = DotPredictor().to(device)
elif args.pred == 'Lorentz':
    pred = LorentzPredictor().to(device)
else:
    raise NotImplementedError

embedding = torch.nn.Embedding(graph.num_nodes(), args.emb_dim).to(device)
if args.init == 'uniform':
    torch.nn.init.uniform_(embedding.weight)
elif args.init == 'ones':
    torch.nn.init.ones_(embedding.weight)
elif args.init == 'orthogonal':
    torch.nn.init.orthogonal_(embedding.weight)
graph.ndata['feat'] = embedding.weight

if args.model == 'GCN':
    model = GCN(graph.ndata['feat'].shape[1], args.hidden, args.norm, args.dp4norm, args.drop_edge, args.relu, args.linear, args.prop_step, args.dropout, args.residual, args.concat_skip, args.conv).to(device)
elif args.model == 'GCN_v1':
    model = GCN_v1(graph.ndata['feat'].shape[1], args.hidden, args.norm, args.relu, args.prop_step, args.dropout, args.multilayer, args.conv, args.res, args.gin_aggr).to(device)
else:
    raise NotImplementedError

parameter = itertools.chain(model.parameters(), pred.parameters(), embedding.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)

# PULL Setup
if args.loss == 'pull':
    pull_K = train_pos_edge.size(0) # Initial K = |E_train|
    # Top-100 Degree Nodes for Candidate Set
    degs = graph.in_degrees()
    top_deg_nodes = torch.topk(degs, 100).indices.cpu().numpy()
    pseudo_pos_dict = {}
    pull_updates = 0

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

print(f'number of parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in pred.parameters()) + sum(embedding.parameters())}')

for epoch in range(1, args.epochs + 1):
    
    # PULL Outer Loop (Update Pseudo-Labels)
    if args.loss == 'pull' and epoch % args.pull_interval == 0 and pull_updates < 10:
        pull_updates += 1
        print(f"PULL Update {pull_updates}: Generating pseudo-labels...")
        
        # 1. Update K
        pull_K = int(pull_K + args.pull_r * train_pos_edge.size(0))
        
        # 2. Generate Candidate Negatives
        # Candidates: Edges where at least one node is in top_deg_nodes
        # We sample random negatives and filter.
        # Efficient Sampling: 5 * K candidates
        num_candidates = 5 * pull_K 
        
        # We need a custom sampling strategy or just reuse neg_sampler?
        # neg_sampler generates negatives for specific positive edges (pos_edge.t()[0])
        # We want global negatives involving top nodes.
        
        # Crude approach: Uniform sample large batch, then filter.
        model.eval()
        with torch.no_grad():
            h = model(graph, graph.ndata['feat'])
            
            # Simple candidate generation:
            # Pair top nodes with random nodes.
            # top_deg_nodes (100) x Random (N)
            # We want ~5*K candidates.
            
            # Sampling: 
            # src = random.choice(top_deg_nodes)
            # dst = random.choice(all_nodes)
            
            candidates_src = np.random.choice(top_deg_nodes, num_candidates)
            candidates_dst = np.random.randint(0, graph.num_nodes(), num_candidates)
            
            # Filter valid (not in graph) - Assume simplified check or rely on massive sparsity
            # For exactness, check has_edges_between
            cand_edges = (torch.tensor(candidates_src).to(device), torch.tensor(candidates_dst).to(device))
            
            # Check existence
            # dgl has_edges_between matches src, dst elementwise
            exists = graph.has_edges_between(cand_edges[0], cand_edges[1])
            # also undirected check? (v, u)
            exists_rev = graph.has_edges_between(cand_edges[1], cand_edges[0])
            valid_mask = ~(exists | exists_rev)
            
            valid_src = candidates_src[valid_mask.cpu().numpy()]
            valid_dst = candidates_dst[valid_mask.cpu().numpy()]
            
            if len(valid_src) == 0:
                print("Warning: No valid candidates found for PULL.")
                pseudo_pos_dict = {}
            else:
                # Predict Scores
                # Batch prediction
                cand_h_src = h[valid_src]
                cand_h_dst = h[valid_dst]
                cand_scores = pred(cand_h_src, cand_h_dst).sigmoid().cpu().numpy() # Use sigmoid for probability
                
                # Take Top-K
                k_actual = min(len(cand_scores), pull_K)
                top_indices = np.argpartition(cand_scores, -k_actual)[-k_actual:]
                
                pseudo_pos_dict = {}
                for idx in top_indices:
                    u, v = int(valid_src[idx]), int(valid_dst[idx])
                    score = float(cand_scores[idx])
                    if u > v: u, v = v, u # Normalize
                    pseudo_pos_dict[(u, v)] = score
                
                print(f"PULL Update: Selected {len(pseudo_pos_dict)} pseudo-positives (Target K={pull_K})")

    if args.loss == 'pull':
        loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred, pseudo_pos_dict)
    else:
        loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred)
    losses.append(loss)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
    valid_results, train_results, val_loss = eval(model, graph, train_pos_edge, valid_pos_edge, valid_neg_edge, pred)
    valid_list.append(valid_results[args.metric])
    
    # Store validation loss
    if 'val_losses' not in locals():
        val_losses = []
    val_losses.append(val_loss)
    valid_list.append(valid_results[args.metric])
    for k, v in valid_results.items():
        print(f'Validation {k}: {v:.4f}')
    for k, v in train_results.items():
        print(f'Train {k}: {v:.4f}')
    if args.dataset == 'ogbl-collab':
        graph_t = graph.clone()
        u, v = valid_pos_edge.t()
        graph_t.add_edges(u, v)
        graph_t.add_edges(v, u)
    else:
        graph_t = graph
    test_results = test(model, graph_t, test_pos_edge, test_neg_edge, pred)
    test_list.append(test_results[args.metric])
    for k, v in test_results.items():
        print(f'Test {k}: {v:.4f}')

    if args.dataset == 'ogbl-ppa':
        if best_epoch + 200 < epoch:
            break
    if args.dataset == 'ogbl-ddi':
        if train_results[args.metric] > 0.90:
            break

    if valid_results[args.metric] > best_val:
        if args.dataset == 'ogbl-ddi':
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results
            torch.save({'model': model.state_dict(), 'pred': pred.state_dict(), 'embedding': embedding.state_dict() if embedding is not None else None}, best_model_path)

        elif args.dataset != 'ogbl-ddi':
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results
            torch.save({'model': model.state_dict(), 'pred': pred.state_dict(), 'embedding': embedding.state_dict() if embedding is not None else None}, best_model_path)


    print(f"Epoch {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train hit: {train_results[args.metric]:.4f}, Valid hit: {valid_results[args.metric]:.4f}, Test hit: {test_results[args.metric]:.4f}")
    wandb.log({'loss': loss, 'val_loss': val_loss, 'train_hit': train_results[args.metric], 'valid_hit': valid_results[args.metric], 'test_hit': test_results[args.metric]})

# plot_dot_product_dist(graph.ndata['feat'])
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


