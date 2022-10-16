import dgl
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from model import GraphTL
from aug import aug
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
#from visualization import plot_embeddings
import torch.multiprocessing as mp
from itertools import zip_longest
import torch.nn.functional as F
def fit_ppi_linear(num_classes, train_data, val_data, test_data, device, repeat=1):
    r"""
        Trains a linear layer on top of the representations. This function is specific to the PPI dataset,
        which has multiple labels.
        """

    def test(classifier, data,evaluator):
        #evaluator = Evaluator(name='ogbn-arxiv')
        classifier.eval()
        x, label = data
        label = label.to(device)
        # feed to network and classifier
        pred_logits = classifier(x.to(device))
        y_pred = pred_logits.argmax(dim=-1, keepdim=True)
        test_acc = evaluator.eval({
            'y_true': label,
            'y_pred': y_pred,
        })['acc']
        return test_acc

    num_feats = train_data[0].size(1)
    #criterion = torch.nn.BCEWithLogitsLoss()

    # normalization
    # mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(0, unbiased=False, keepdim=True)
    # train_data[0] = (train_data[0] - mean) / std
    # val_data[0] = (val_data[0] - mean) / std
    # test_data[0] = (test_data[0] - mean) / std

    best_val_f1 = []
    test_f1 = []
    for _ in range(repeat):
        tmp_best_val_f1 = 0
        tmp_test_f1 = 0
        for weight_decay in 2.0 ** np.arange(-10, 1, 2):
            classifier = torch.nn.Linear(num_feats, num_classes,device=device)
            optimizer1 = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)
            #train(classifier, train_data, optimizer,criterion)
            evaluator = Evaluator(name='ogbn-arxiv')
            x, label = train_data
            x, label = x.to(device), label.to(device)
            for step in range(1000):
                classifier.train()
                # forward
                pred_logits = classifier(x)
                pred_logits=pred_logits.log_softmax(dim=-1)
                l = F.nll_loss(pred_logits, label.squeeze(1))
                y_pred = pred_logits.argmax(dim=-1, keepdim=True)
                test_acc = evaluator.eval({
                    'y_true': label,
                    'y_pred': y_pred,
                })['acc']
                # print(0,test_acc)
                optimizer1.zero_grad()
                l.backward()
                optimizer1.step()
            val_f1 = test(classifier, val_data,evaluator)
            if val_f1 > tmp_best_val_f1:
                tmp_best_val_f1 = val_f1
                tmp_test_f1 = test(classifier, test_data,evaluator)
        best_val_f1.append(tmp_best_val_f1)
        test_f1.append(tmp_test_f1)
    return [best_val_f1], [test_f1]
def train(graph, feat, feat_, train_mask,val_mask, test_mask,labels):
    # torch.cuda.set_device(rank)
    # dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)
    graph_2 = graph.add_self_loop()
    graph.remove_self_loop()
    in_dim = feat.shape[1]
    train_nids = range(feat.shape[0])
    model = GraphTL(in_dim, 512, 1024, 3, 1.0).cuda()
    #model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005, weight_decay=5e-6)
    #graph_2.ndata['label'] = labels
    graph_2.ndata['feat'] = feat
    graph_2.ndata['feat_'] = feat_
    pos=[]
    for epoch in range(7):
        model.train()
        graph1, feat1 = aug(graph, feat, 0.2, 0.9)
        graph1.ndata['feat']=feat1

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
        train_dataloader = dgl.dataloading.NodeDataLoader(graph_2, indices=train_nids, batch_size=1024*5, graph_sampler=sampler,device='cuda', use_ddp=False)
        sampler1 = dgl.dataloading.MultiLayerFullNeighborSampler(3)
        train_dataloader1 = dgl.dataloading.NodeDataLoader(graph1, indices=train_nids, batch_size=1024*5, graph_sampler=sampler1,device='cuda',use_ddp=False)
        # train_dataloader.set_epoch(epoch)
        # train_dataloader1.set_epoch(epoch)
        i=0
        # torch.distributed.barrier()
        for (input_nodes1, output_nodes1, blocks1),(input_nodes, output_nodes, blocks) in zip_longest(train_dataloader1,train_dataloader):

            x = blocks[0].srcdata['feat']
            f = blocks[2].dstdata['feat_']
            x1 = blocks1[0].srcdata['feat']
            output_nodes = output_nodes.to('cpu')
            input_nodes = input_nodes.to('cpu')
            output_nodes1 = output_nodes1.to('cpu')
            input_nodes1 = input_nodes1.to('cpu')
            if epoch==0:
                a = output_nodes.size(0)
                p = torch.zeros([a, 169343], dtype=torch.float,device='cpu')
                for _ in range(20):
                    walks = dgl.sampling.random_walk(graph, output_nodes, length=2)[0]
                    for i in range(a):
                        for j in range(3):
                            t = walks[i][j]
                            p[i][t] = p[i][t] + 1
                p = F.normalize(p)
                p = torch.matmul(p, p.T)
                pos.append(p)
            else:
                p=pos[i]
                i=i+1
            device=x.device
            p=p.to(device)
            loss = model(blocks1, blocks, x1, x, p, f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')
        if epoch % 1 == 0:
            model1 = model.to('cpu')
            model.eval()
            model1.eval()
            # train_reps = []
            # train_labels = []
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
            train_dataloader = dgl.dataloading.NodeDataLoader(graph_2, indices=train_nids, batch_size=2048000,
                                                      graph_sampler=sampler,
                                                      device='cpu', use_ddp=False)
            for input_nodes, output_nodes, blocks in train_dataloader:
                x = blocks[0].srcdata['feat']
                embeds = model1.get_embedding(blocks, x)
                # train_reps.append(embeds)
                # train_labels.append(blocks[2].dstdata['label'])
            # train_reps = torch.cat(train_reps, dim=0)
            # train_labels = torch.cat(train_labels, dim=0)
            train_datas = [embeds[train_mask], labels[train_mask]]
            val_datas = [embeds[val_mask], labels[val_mask]]
            test_datas = [embeds[test_mask], labels[test_mask]]
            #num_classes = train_datas[1].shape[1]
            val_scores, test_scores = fit_ppi_linear(40, train_datas, val_datas, test_datas, device, 3)
            print('Epoch: {:04d} | Best Val F1: {:.4f} | Test F1: {:.4f}'.format(epoch, np.mean(val_scores),
                                                                                 np.mean(test_scores)))
            model.to(device)
    print("=== Final ===")
    return model
if __name__ == '__main__':
    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]
    feat = graph.ndata['feat']
    graph.ndata['label']=labels
    graph = dgl.to_bidirected(graph)
    train_mask = split_idx['train']
    val_mask = split_idx['valid']
    test_mask = split_idx['test']
    train_nids = range(feat.shape[0])
    feat_ = F.normalize(feat)
    n_procs = torch.cuda.device_count()
    model=train(graph, feat, feat_, train_mask, val_mask, test_mask, labels)