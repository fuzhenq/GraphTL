import torch as th
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class layer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 bias=True):
        super(layer, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, in_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, in_feats)))
        self.attn_l1 = nn.Parameter(th.FloatTensor(size=(1, num_heads, in_feats)))
        self.attn_r1 = nn.Parameter(th.FloatTensor(size=(1, num_heads, in_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_l1, gain=gain)
        nn.init.xavier_normal_(self.attn_r1, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                print("eroer")
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                if graph.is_block:
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
                s = h_src.repeat(1,self._num_heads)
                s = s.reshape(h_src.shape[0],self._num_heads,h_src.shape[1])
                #s = F.normalize(s,dim=-1)
                d = h_dst.repeat(1,self._num_heads)
                d = d.reshape(h_dst.shape[0],self._num_heads,h_dst.shape[1])
                #d = F.normalize(d, dim=-1)
                el = (s * self.attn_l)
                er = (d * self.attn_r)
                #graph.srcdata.update({'ft': feat_src, 'el': el})
                graph.srcdata.update({'el': el})
                graph.dstdata.update({'er': er})
                el = (s * self.attn_l1)
                er = (d * self.attn_r1)
                #print(0,el.size())
                graph.srcdata.update({'el1': el})
                graph.dstdata.update({'er1': er})
                graph.apply_edges(fn.u_dot_v('el', 'er', 'e1'))
                e1 = graph.edata.pop('e1')
                graph.apply_edges(fn.u_dot_v('el1', 'er1', 'e2'))
                e2 = graph.edata.pop('e2')
                el=el.pow(2).sum(dim=-1,keepdims=True)
                #print(1,el.size())
                er = er.pow(2).sum(dim=-1, keepdims=True)
                graph.srcdata.update({'el2': el})
                graph.dstdata.update({'er2': er})
                graph.apply_edges(fn.u_add_v('el2', 'er2', 'e3'))
                el.to('cpu')
                er.to('cpu')
                #e = self.leaky_relu(graph.edata.pop('e'))
                e3 = graph.edata.pop('e3')
                e = e1 - 2*e2 + e3
                # compute softmax
                graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                # message passing
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
            graph.srcdata.update({'ft': feat_src})

            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            #rst = F.elu(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst