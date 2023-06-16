"""Hyperbolic layers."""
import math
from numpy import dtype
from sklearn import manifold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt
from kernels.kernel_points import load_kernels

from geoopt import ManifoldParameter

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias, scale=10)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

class LorentzMultiHeadedAttention(nn.Module):
    """
    Hyperbolic Multi-headed Attention
    """

    def __init__(self, head_count, dim, manifold, dropout = 0.0):
        super(LorentzMultiHeadedAttention, self).__init__()
        self.dim_per_head = dim // head_count
        self.dim = dim
        self.manifold = manifold
        self.head_count = head_count

        self.linear_key = LorentzLinear(manifold, dim, dim, dropout=dropout)
        self.linear_value = LorentzLinear(manifold, dim, dim, dropout=dropout)
        self.linear_query = LorentzLinear(manifold, dim, dim, dropout=dropout)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(dim)]))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, key, value, query, mask = None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        # key_len = key.size(1)
        # query_len = query.size(1)

        def shape(x):
            """Projection."""
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, head_count, dim_per_head)
            return x.transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).view(batch_size, -1, head_count * dim_per_head)

        query = self.linear_query(query)
        key = self.linear_key(key)
        value =  self.linear_value(value)
        key = shape(key)
        value = shape(value)
        query = shape(query)
        # key_len = key.size(2)
        # query_len = query.size(2)

        attn = (2 + 2 * self.manifold.cinner(query, key)) / self.scale + self.bias
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            attn = attn.masked_fill(mask, -1e18)
        context = self.manifold.mid_point(value, attn)
        context = unshape(context)

        return context

class LorentzCentroidDistance(nn.Module):
    """
    Hyerbolic embeddings to Euclidean distance
    """

    def __init__(self, dim, n_classes, manifold, bias = True):
        super(LorentzCentroidDistance, self).__init__()
        self.manifold = manifold
        self.input_dim = dim
        self.output_dim = n_classes
        self.use_bias = bias
        self.cls = ManifoldParameter(
            self.manifold.random_normal((n_classes, dim), std=1./math.sqrt(dim)), 
            manifold=self.manifold)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_classes))
        
    def forward(self, x):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias

class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, local_agg, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att, local_agg)
        # self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        # h = self.hyp_act.forward(h)
        output = h, adj
        return output


class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        # print(type(self.in_features), type(self.out_features), self.bias)
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1 + 1e-4
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            # self.att = DenseAtt(in_features, dropout)
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        # x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                # x_local_tangent = []
                # # for i in range(x.size(0)):
                # #     x_local_tangent.append(self.manifold.logmap(x[i], x))
                # # x_local_tangent = torch.stack(x_local_tangent, dim=0)
                # x_local_tangent = self.manifold.clogmap(x, x)
                # # import pdb; pdb.set_trace()
                # adj_att = self.att(x, adj)
                # # att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                # support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                # output = self.manifold.expmap(x, support_t)
                # return output
                query = self.query_linear(x)
                key = self.key_linear(x)
                att_adj = 2 + 2 * self.manifold.cinner(query, key)
                att_adj = att_adj / self.scale + self.bias
                att_adj = torch.sigmoid(att_adj)
                att_adj = torch.mul(adj.to_dense(), att_adj)
                support_t = torch.matmul(att_adj, x)
            else:
                adj_att = self.att(x, adj)
                support_t = torch.matmul(adj_att, x)
        else:
            support_t = torch.spmm(adj, x)
        # output = self.manifold.expmap0(support_t, c=self.c)
        denom = (-self.manifold.inner(None, support_t, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        output = support_t / denom
        return output

    def attention(self, x, adj):
        pass


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
    
    
    
    
def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')

class KernelPointAggregation(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, KP_extent, radius,
                 manifold, use_bias, dropout, nonlin=None,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        super(KernelPointAggregation, self).__init__()
        # Save parameters
        self.manifold = manifold
        self.K = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        
        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
#         self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
#                                  requires_grad=True)
        self.linears = nn.ModuleList([LorentzLinear(manifold, in_channels, out_channels, use_bias, dropout, nonlin=nonlin)
                                      for _ in range(self.K)])

        # Initiate weights for offsets
        if deformable:
            self.offset_dim = (self.in_channels - 1) * self.K + 1
            self.offset_conv = KernelPointAggregation(self.K,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      self.manifold,
                                      use_bias,
                                      dropout,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            # self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)
        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def init_KP(self):
        # Create one kernel disposition. Choose the KP distance to center thanks to the KP extent
        K_tangent = load_kernels(manifold = self.manifold, 
                                 radius = self.KP_extent, 
                                 num_kpoints = self.K, 
                                 dimension = self.in_channels)

        return nn.Parameter(K_tangent, requires_grad=False)
    
    def get_kernel_pos(self, x, nei, nei_mask, sample, sample_num, transp, radius = None):
        n, d = x.shape
        if radius == None:
            radius = self.KP_extent
        if not transp:
            res = self.manifold.expmap0(self.kernel_points).repeat(n, 1, 1)
            # print(res)
        else:
            x_k = x.repeat(1, 1, self.K - 1).view(n, self.K - 1, d) # (n, k-1, d)
            tmp = self.manifold.transp0(x_k, self.kernel_points[1:]) # parallel transport to x
            tmp = self.manifold.expmap(x_k, tmp) # expmap to manifold
            res = torch.concat((tmp, x.view(n, 1, d)), 1) # add fixed kernel (n, k, d)
        if self.deformable:
            offset = self.offset_conv(x, nei, nei_mask, sample, sample_num) # (n, (d - 1) * k + 1)
            # print(offset)
            offset = self.manifold.split(offset, self.K) # (n, k, d)
            dis = self.manifold.dist0(offset).max()
            offset = self.manifold.logmap0(offset)
            # print(offset)
            offset *= radius / dis
            offset = self.manifold.transp0(res, offset)
            # print(offset)
            res = self.manifold.expmap(res, offset)
        return res
    
    def get_nei_kernel_dis(self, x_kernel, x_nei):
        n, nei_num, d = x_nei.shape

        # x_nei_k = x_nei.repeat(1, 1, 1, self.K).view(n, nei_num, self.K, d) # (n, nei_num, k, d)
        # x_nei_k = x_nei_k.swapaxes(1, 2) # (n, k, nei_num, d)
        # x_kernel_nei = x_kernel.repeat(1, 1, 1, nei_num).view(n, self.K, nei_num, d) # (n, k, nei_num, d)
        # return self.manifold.dist(x_nei_k, x_kernel_nei) # (n, k, nei_num)

        return self.manifold.dist(x_nei.repeat(1, 1, 1, self.K).view(n, nei_num, self.K, d).swapaxes(1, 2), x_kernel.repeat(1, 1, 1, nei_num).view(n, self.K, nei_num, d))

    def transport_x(self, x, x_nei):
        # x = x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape)
        # x_tan_nei = self.manifold.logmap(x, x_nei) 
        # x0_tan_nei = self.manifold.transp0back(x, x_tan_nei) 
        # x0_nei = self.manifold.expmap0(x0_tan_nei)
        # x0 = self.manifold.origin(x.shape[-1]).repeat(x.shape[0], 1)

        x0_nei = self.manifold.expmap0(self.manifold.transp0back(x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape), self.manifold.logmap(x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape), x_nei) ) )
        x0 = self.manifold.origin(x.shape[-1]).repeat(x.shape[0], 1)
        return x0, x0_nei, x
    
    def apply_kernel_transform(self, x_nei):
        res = []
        for k in range(self.K):
            res.append(self.linears[k](x_nei).unsqueeze(1))
        return torch.concat(res, dim = 1)
    
    def avg_kernel(self, x_nei_transform, x_nei_kernel_dis):
        x_nei_transform = x_nei_transform.swapaxes(1, 2) # (n, nei_num, k, d')
        x_nei_kernel_dis = x_nei_kernel_dis.swapaxes(1, 2).unsqueeze(3) # (n, nei_num, k)
        return self.manifold.mid_point(x_nei_transform, x_nei_kernel_dis.swapaxes(2, 3)) # (n, nei_num, d')

    def sample_nei(self, nei, nei_mask, sample_num):
        new_nei = []
        new_nei_mask = []
        for i in range(len(nei)):
            tot = nei_mask[i].sum()
            if tot > 0:
                new_nei.append(nei[i][torch.randint(0, tot, (sample_num,))])
                new_nei_mask.append(torch.ones((sample_num,), device=nei.device))
            else:
                new_nei.append(torch.zeros((sample_num,), device=nei.device))
                new_nei_mask.append(torch.zeros((sample_num,), device=nei.device))
        return torch.stack(new_nei).type(torch.long), torch.stack(new_nei_mask).type(torch.long)
        
    def forward(self, x, nei, nei_mask, transp = True, sample = False, sample_num = 16):
        # x (n, d) data value
        # nei (n, nei_num) neighbors
        # nei_mask (n, nei_num) 0/1 mask for neighbors
        if sample:
            nei, nei_mask = self.sample_nei(nei, nei_mask, sample_num)
        
        x_nei = gather(x, nei) # (n, nei_num, d)
        if transp:
            x, x_nei, x0 = self.transport_x(x, x_nei)
        n, nei_num, d = x_nei.shape
        x_kernel = self.get_kernel_pos(x, nei, nei_mask, sample, sample_num, transp) # (n, k, d)
        x_nei_kernel_dis = self.get_nei_kernel_dis(x_kernel, x_nei) # (n, k, nei_num)
        nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num) # (n, k, nei_num)
        x_nei_kernel_dis = x_nei_kernel_dis * nei_mask
        x_nei_transform = self.apply_kernel_transform(x_nei) # (n, k, nei_num, d')
        x_nei_transform = self.avg_kernel(x_nei_transform, x_nei_kernel_dis).squeeze(2) # (n, nei_num, d')
        x_final = self.manifold.mid_point(x_nei_transform) # (n, d')
        return x_final

class KPGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, kernel_size, KP_extent, radius, in_features, out_features, use_bias, dropout, nonlin=None, deformable = False):
        super(KPGraphConvolution, self).__init__()
        self.net = KernelPointAggregation(kernel_size, in_features, out_features, KP_extent, radius, manifold, use_bias, dropout, nonlin=nonlin, deformable=deformable)

    def forward(self, input):
        x, nei, nei_mask = input
        h = self.net(x, nei, nei_mask)
        output = h, nei, nei_mask
        return output

class KernelPointMidPoint(nn.Module):
    def __init__(self, manifold, kernel_size, KP_extent, radius, in_features, out_features, use_bias, dropout, nonlin=None):
        super(KernelPointMidPoint, self).__init__()
        self.manifold = manifold 
        self.net = KernelPointAggregation(kernel_size, in_features, out_features, KP_extent, radius, manifold, use_bias, dropout, nonlin=nonlin)

    def foward(self, x):
        x0 = self.manifold(x)
        return self.net(x0, x, torch.zeros(x.shape[:-1]).to(x.device()))
