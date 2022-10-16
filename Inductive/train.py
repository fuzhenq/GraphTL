import copy
import time

import dgl
from dgl.dataloading import GraphDataLoader
from sklearn import metrics

from model import GraphTL
from aug import aug
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import warnings
import deepwalk
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import zip_longest
from dgl.data import PPIDataset

def fit_ppi_linear(num_classes, train_data, val_data, test_data, device, repeat=1):
    r"""
        Trains a linear layer on top of the representations. This function is specific to the PPI dataset,
        which has multiple labels.
        """

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.data.cpu().numpy()

        # feed to network and classifier
        pred_logits = classifier(x.to(device))
        predict = np.where(pred_logits.data.cpu().numpy() >= 0., 1, 0)

        return metrics.f1_score(label, predict, average='micro')

    num_feats = train_data[0].size(1)
    criterion = torch.nn.BCEWithLogitsLoss()

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(0, unbiased=False, keepdim=True)
    train_data[0] = (train_data[0] - mean) / std
    val_data[0] = (val_data[0] - mean) / std
    test_data[0] = (test_data[0] - mean) / std

    best_val_f1 = []
    test_f1 = []
    for _ in range(repeat):
        tmp_best_val_f1 = 0
        tmp_test_f1 = 0
        for weight_decay in 2.0 ** np.arange(-10, 11, 2):
            classifier = torch.nn.Linear(num_feats, num_classes,device=device)
            optimizer1 = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)
            x, label = train_data
            x, label = x.to(device), label.to(device)
            for step in range(300):
                classifier.train()
                # forward
                pred_logits = classifier(x)
                l = criterion(pred_logits, label)
                optimizer1.zero_grad()
                l.backward()
                optimizer1.step()
            val_f1 = test(classifier, val_data)
            if val_f1 > tmp_best_val_f1:
                tmp_best_val_f1 = val_f1
                tmp_test_f1 = test(classifier, test_data)
        best_val_f1.append(tmp_best_val_f1)
        test_f1.append(tmp_test_f1)
    return [best_val_f1], [test_f1]

def train(train_val_dataset, train_data,val_data,test_data):
    in_dim=train_val_dataset[0].ndata['feat'].size(1)
    model = GraphTL(in_dim, 256, 512, 3, 1.0).to('cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(451):
        model.train()
        for idx, subgraph in enumerate(train_val_dataset):
            x = subgraph.ndata['feat'].float()
            train_nids = range(x.shape[0])
            subgraph1, x1 = aug(subgraph, x, 0.2, 0.4)
            subgraph1.ndata['feat'] = x1
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
            train_dataloader = dgl.dataloading.NodeDataLoader(subgraph, nids=train_nids, batch_size=20000, block_sampler=sampler,device='cuda:0', use_ddp=False)
            sampler1 = dgl.dataloading.MultiLayerFullNeighborSampler(3)
            train_dataloader1 = dgl.dataloading.NodeDataLoader(subgraph1, nids=train_nids, batch_size=20000, block_sampler=sampler1,device='cuda:0',use_ddp=False)
            for (input_nodes1, output_nodes1, blocks1), (input_nodes, output_nodes, blocks) in zip_longest(train_dataloader1, train_dataloader):
                x = blocks[0].srcdata['feat']
                p = blocks[2].dstdata['pos_']
                f = blocks[2].dstdata['feat_']
                x1 = blocks1[0].srcdata['feat']
                loss = model(blocks1, blocks, x1, x, p, f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #subgraph.remove_self_loop()
        print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')
        if epoch % 50 == 0:
            model.eval()
            train_reps = []
            train_labels = []
            for idx, subgraph in enumerate(train_data):
                x = subgraph.ndata['feat'].float()
                train_nids = range(x.shape[0])
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
                train_dataloader = dgl.dataloading.NodeDataLoader(subgraph, nids=train_nids, batch_size=1000000,
                                                                  block_sampler=sampler, device='cuda:0', use_ddp=False)
                for input_nodes, output_nodes, blocks in train_dataloader:
                    x = blocks[0].srcdata['feat']

                    embeds = model.get_embedding(blocks,x)
                    train_reps.append(embeds)
                    train_labels.append(blocks[2].dstdata['label'])
            train_reps = torch.cat(train_reps, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            train_datas = [train_reps, train_labels]

            val_reps = []
            val_labels = []
            for idx, subgraph in enumerate(val_data):
                x = subgraph.ndata['feat'].float()
                train_nids = range(x.shape[0])
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
                train_dataloader = dgl.dataloading.NodeDataLoader(subgraph, nids=train_nids, batch_size=100000,
                                                                  block_sampler=sampler, device='cuda:0', use_ddp=False)
                for input_nodes, output_nodes, blocks in train_dataloader:
                    x = blocks[0].srcdata['feat']
                    embeds = model.get_embedding(blocks,x)
                    val_reps.append(embeds)
                    val_labels.append(blocks[2].dstdata['label'])
            val_reps = torch.cat(val_reps, dim=0)
            val_labels = torch.cat(val_labels, dim=0)
            val_datas = [val_reps, val_labels]
            test_reps = []
            test_labels = []
            for idx, subgraph in enumerate(test_data):
                x = subgraph.ndata['feat'].float()
                train_nids = range(x.shape[0])
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
                train_dataloader = dgl.dataloading.NodeDataLoader(subgraph, nids=train_nids, batch_size=100000,
                                                                  block_sampler=sampler,
                                                                  device='cuda:0', use_ddp=False)
                for input_nodes, output_nodes, blocks in train_dataloader:
                    x = blocks[0].srcdata['feat']

                    embeds = model.get_embedding(blocks,x)
                    test_reps.append(embeds)
                    test_labels.append(blocks[2].dstdata['label'])
            test_reps = torch.cat(test_reps, dim=0)
            test_labels = torch.cat(test_labels, dim=0)
            test_datas = [test_reps, test_labels]
            num_classes = train_datas[1].shape[1]
            val_scores, test_scores = fit_ppi_linear(num_classes, train_datas, val_datas, test_datas, 'cuda:1',
                                                     3)
            print('Epoch: {:04d} | Best Val F1: {:.4f} | Test F1: {:.4f}'.format(epoch, np.mean(val_scores),
                                                                                 np.mean(test_scores)))
    print("=== Final ===")
    return model
if __name__ == '__main__':
    train_dataset = PPIDataset(mode='train')
    val_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    train_val_dataset = [i for i in train_dataset] + [i for i in val_dataset]
    for idx, data in enumerate(train_val_dataset):
        adj = data.adjacency_matrix_scipy().todense()
        nx_g = deepwalk.load_edgelist(np.asarray(adj), directed=True)
        # path_length stands for t+1 in the paper
        pos_ = deepwalk.build_deepwalk_corpus(nx_g, num_paths=50, path_length=3, alpha=0, nb_nod=adj.shape[0])
        pos_ = torch.FloatTensor(pos_)#Random walk to get the result T
        pos_ = F.normalize(pos_)
        data.ndata['pos_'] = pos_
        feat_ = F.normalize(data.ndata['feat'])
        data.ndata['feat_'] = feat_
    train_dataset = PPIDataset(mode='train')
    val_dataset = PPIDataset(mode='valid')
    train(train_val_dataset, train_dataset,val_dataset,test_dataset)