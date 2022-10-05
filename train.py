import dgl
from model import GraphTL
from aug import aug
import argparse
from dataset import load
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from eval import label_classification
import deepwalk
import torch.distributed as dist
from visualization import plot_embeddings
import torch.multiprocessing as mp
from itertools import zip_longest

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cora')
parser.add_argument('--pathlength', type=int, default=2,help='Neighborhood range of topological similarity')
parser.add_argument('--numpath', type=int, default=50,help='Number of paths used to calculate topological similarity')
parser.add_argument('--epochs', type=int, default=350, help='Number of training periods.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1.5e-4, help='Weight decay.')
parser.add_argument("--hid_dim", type=int, default=256, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')
parser.add_argument("--batchsize",type=int,default=1000000,help='Default is large enough')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument('--der', type=float, default=0.5, help='Drop edge ratio of the augmented augmentation.')
parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio of the augmented augmentation.')
parser.add_argument('--walk-length', type=int, default=2, help='Length of walk per source. Default is 20.')
parser.add_argument('--num-walks', type=int, default=2, help='Number of walks per source. Default is 50.')
parser.add_argument('--numheads',type=int,default=2)
parser.add_argument('--K',type=int,default=50,help='Number of similar nodes per node')
args = parser.parse_args()

def train(rank, world_size, graph, feat, pos_, feat_, train_mask, test_mask,labels):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)
    graph_2 = graph.add_self_loop()
    in_dim = feat.shape[1]
    train_nids = range(feat.shape[0])

    model = GraphTL(in_dim, args.hid_dim, args.out_dim, args.num_layers,args.numheads,args.K).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    graph_2.ndata['feat'] = feat
    graph_2.ndata['pos_'] = pos_
    graph_2.ndata['feat_'] = feat_

    for epoch in range(args.epochs):
        model.train()
        graph1, feat1 = aug(graph, feat, args.dfr, args.der)
        graph1.ndata['feat']=feat1
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
        train_dataloader = dgl.dataloading.NodeDataLoader(graph_2, nids=train_nids, batch_size=args.batchsize, block_sampler=sampler,device='cuda', use_ddp=True)
        sampler1 = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
        train_dataloader1 = dgl.dataloading.NodeDataLoader(graph1, nids=train_nids, batch_size=args.batchsize, block_sampler=sampler1,device='cuda',use_ddp=True)
        train_dataloader.set_epoch(epoch)
        train_dataloader1.set_epoch(epoch)
        for (input_nodes1, output_nodes1, blocks1),(input_nodes, output_nodes, blocks) in zip_longest(train_dataloader1,train_dataloader):
            x = blocks[0].srcdata['feat']
            p = blocks[1].dstdata['pos_']
            f = blocks[1].dstdata['feat_']
            x1 = blocks1[0].srcdata['feat']
            loss = model(blocks1, blocks, x1, x, p, f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')
    graph_2.ndata.pop('pos_')
    graph_2.ndata.pop('feat_')
    model.eval()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    train_dataloader = dgl.dataloading.NodeDataLoader(graph_2, nids=train_nids, batch_size=args.batchsize,
                                                      block_sampler=sampler,
                                                      device='cpu', use_ddp=False)
    for input_nodes, output_nodes, blocks in train_dataloader:
        x = blocks[0].srcdata['feat']
        model = model.module.to('cpu')
        embeds = model.get_embedding(blocks, x)
        print(embeds.size())
    #embeds = model.get_embedding(graph.add_self_loop(), feat)
    plot_embeddings(embeddings=embeds.cpu().numpy(), label_file=labels.cpu().numpy())
    label_classification(embeds.cpu(), labels.cpu(), train_mask.cpu(), test_mask.cpu(), split='random')
    print("=== Final ===")
    return model
if __name__ == '__main__':
    graph, feat, labels, train_mask, test_mask = load(args.dataname) #Load the dataset
    graph.create_formats_()
    adj = graph.adjacency_matrix_scipy().todense()
    nx_g = deepwalk.load_edgelist(np.asarray(adj), directed=True)
    pos_ = deepwalk.build_deepwalk_corpus(nx_g, num_paths=args.numpath, path_length=args.pathlength, alpha=0,#path_length stands for t+1 in paper
                                          nb_nod=adj.shape[0]) #random walk to get the result T in paper
    pos_ = torch.FloatTensor(pos_)
    pos_ = F.normalize(pos_)
    feat_ = F.normalize(feat)
    in_dim = feat.shape[1]
    n_procs = 1 #Number of GPUs
    model = mp.spawn(train, args=(n_procs, graph, feat, pos_, feat_, train_mask, test_mask,labels), nprocs=n_procs)
