import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from Layer import layer
#from python.dgl.nn.pytorch.conv.gatconv import GATConv

class Encoder(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.norms = nn.ModuleList()
        if num_layers > 1:
        # input projection (no residual)
            self.gat_layers.append(layer(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, True, None))
            self.norms.append(nn.BatchNorm1d(num_hidden * heads[0]))
            # hidden layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(layer(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, True, None))
                self.norms.append(nn.BatchNorm1d(num_hidden * heads[l-1]))
            # output projection
            self.gat_layers.append(layer(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, True, None))
            #self.norms.append(nn.BatchNorm1d(num_classes * heads[-1]))
        else:
            self.gat_layers.append(layer(
                in_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g[l], h)
            # h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
            if l != self.num_layers - 1:
                h = h.flatten(1)
                h=self.norms[l](h)
                h=F.elu(h,inplace=True)
            else:

                #h = self.norms[-1](h)
                h = F.elu(h, inplace=True)
                h = h.mean(1)
        return h
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        z = self.fc1(x)
        return z


class GraphTL(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, temp):
        super(GraphTL, self).__init__()
        heads = ([4] * (num_layers - 1)) + [6]
        self.encoder = Encoder(num_layers=num_layers,in_dim=in_dim, num_hidden=hid_dim, num_classes=out_dim, heads=heads,
                           activation=F.relu, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=True)
        self.proj = MLP(out_dim, out_dim)
        self.alpha = th.nn.Parameter(th.FloatTensor(1), requires_grad=True).cuda()
        self.alpha.data.fill_(0.0).cuda()
        self.temp = temp
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2, pos_, feat_):
        f = lambda x: th.exp(x/self.temp)
        pos_=pos_
        feat_ = feat_
        pos_ = th.matmul(pos_, pos_.T)
        feat_ = th.matmul(feat_, feat_.T)
        pos_ = (F.sigmoid(self.alpha) * pos_ + (1-F.sigmoid(self.alpha)) * feat_)
        top, index = th.topk(pos_, 1, dim=1, sorted=False)
        pos = th.zeros_like(pos_)
        pos = pos.scatter_(1, index, top)
        pos=th.multiply(pos,pos_)
        neg = (1 - pos_)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        # print(refl_sim.size(),neg.size())
        refl_sim = th.multiply(refl_sim, neg)
        loss = -th.log(th.multiply(pos,between_sim).sum(1)/ refl_sim.sum(1))
        return loss

    def get_embedding(self, graph, feat):
        h = self.encoder(graph, feat)
        return h.detach()

    def forward(self, graph1, graph2, feat1, feat2, pos_, feat_):
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)
        z1 = self.proj(h1)
        z2 = self.proj(h2)
        ret = self.get_loss(z2, z1, pos_, feat_)
        return ret.mean()