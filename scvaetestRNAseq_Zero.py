import gc

import numpy as np

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import anndata
import pyro
import pyro.distributions as dist
import scanpy.plotting
import torch
import pandas as pd
import torch.nn as nn
from scipy.sparse import csc_matrix
from scvi import _CONSTANTS
from scvi.module.base import PyroBaseModuleClass
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torch.optim import a
from umap import UMAP
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

class H5ADataSet(Dataset):
    def __init__(self, fname):
        self.data = anndata.read_h5ad(fname)

    def __len__(self):
        return self.data.X.shape[0]

    def __getitem__(self, idx):
        x = csc_matrix(self.data.X[idx])[0].toarray()[0]
        x_tensor = x.astype(np.float32)
        return x_tensor

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Encoder(nn.Module):
    #编码器
    def __init__(self, in_dim, z_dim, nhid, hidden_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, nhid)#图卷积
        self.dropout = dropout
        self.fc1 = nn.Linear(nhid, hidden_dim)#线性层
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()
        
    def forward(self, x,adj):
        x = F.relu(self.gc1(x, adj))#图卷积
        x = F.dropout(x, self.dropout, training=self.training)
        hidden = self.softplus(self.fc1(x))#线性层
        z_loc = self.softplus(self.fc21(hidden))
        z_scale = torch.exp(self.softplus(self.fc22(hidden)))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, in_dim, z_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, in_dim)
        self.fc22 = nn.Linear(hidden_dim, in_dim)

        self.softplus = nn.Softplus()

    def forward(self, z):

        hidden = self.softplus(self.fc1(z))

        count = self.softplus(self.fc21(hidden))
        prob = self.softplus(self.fc22(hidden))


        return count, prob




class VAE(PyroBaseModuleClass):
    def __init__(self, n_input, n_latent, nhid,n_clusters, dropout):
        super().__init__()
        self.n_latent = n_latent
        self.n_input = n_input
        self.n_clusters = n_clusters

        self.decoder = Decoder(n_input, n_latent, 200)
        self.log_theta = torch.nn.Parameter(torch.randn(n_input))
        self.gate_logits = torch.nn.Parameter(torch.randn(n_input))
        self.embeddings = nn.Parameter(torch.randn(self.n_clusters, n_latent) * 0.05, requires_grad=True)
        self.encoder = Encoder(n_input, n_latent,  nhid,200,  dropout)

    def model(self, x,adj):
        pyro.module("decoder", self)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.n_latent)))#正态分布
            z_scale = x.new_ones(torch.Size((x.shape[0], self.n_latent)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))#与guid中同名sample计算KL损失
            count, prob = self.decoder(z)#解码

            k, z_dist, dist_prob = self._get_clusters(z)#细胞类型数先验
            z_q = self._get_embeddings(k)
            count1, prob1 = self.decoder1(z_q)

            x_dist = dist.ZeroInflatedNegativeBinomial(total_count=count,probs=prob, gate_logits=self.gate_logits)#参数重分布
            x_dist1 = dist.ZeroInflatedNegativeBinomial(total_count=count1,probs=prob1, gate_logits=self.gate_logits)

            pyro.sample("obs1", x_dist.to_event(1), obs=x)#重构损失
            pyro.sample("obs2", x_dist1.to_event(1), obs=x)

    def guid(self, x,adj):
        pyro.module("encoder", self)
        with pyro.plate("data", x.shape[0]):
            x_ = torch.log(1+x)
            [qz_m, qz_v] = self.encoder(x_,adj)#编码
            pyro.sample("latent", dist.Normal(qz_m,qz_v).to_event(1))#与model中同名sample计算KL损失



    def getZ(self, x,adj):
        z_loc, z_scale = self.encoder(x,adj)
        z = dist.Normal(z_loc, z_scale).sample()
        k, z_dist, dist_prob = self._get_clusters(z)
        z_q = self._get_embeddings(k)
        return z_q


    def _get_clusters(self, z_e):
        """

        Assign each sample to a cluster based on euclidean distances.

        Parameters
        ----------
        z_e: torch.Tensor
            low-dimensional encodings

        Returns
        -------
        k: torch.Tensor
            cluster assignments
        z_dist: torch.Tensor
            distances between encodings and centroids
        dist_prob: torch.Tensor
            probability of closeness of encodings to centroids transformed by t-distribution

        """

        _z_dist = (z_e.unsqueeze(1) - self.embeddings.unsqueeze(0)) ** 2
        z_dist = torch.sum(_z_dist, dim=-1)


        dist_prob = self._t_dist_sim(z_dist, df=10)
        k = torch.argmax(dist_prob, dim=-1)


        return k, z_dist, dist_prob


    def _get_embeddings(self, k):
        """

        Get the embeddings (discrete representations).

        Parameters
        ----------
        k: torch.Tensor
            cluster assignments

        Returns
        -------
        z_q: torch.Tensor
            low-dimensional embeddings (discrete representations)

        """

        k = k.long()
        _z_q = []
        for i in range(len(k)):
            _z_q.append(self.embeddings[k[i]])

        z_q = torch.stack(_z_q)

        return z_q

    def _t_dist_sim(self, z_dist, df=10):
        """
        Transform distances using t-distribution kernel.

        Parameters
        ----------
        z_dist: torch.Tensor
            distances between encodings and centroids

        Returns
        -------
        dist_prob: torch.Tensor
            probability of closeness of encodings to centroids transformed by t-distribution

        """

        _factor = - ((df + 1) / 2)
        dist_prob = torch.pow((1 + z_dist / df), _factor)
        dist_prob = dist_prob / dist_prob.sum(axis=1).unsqueeze(1)

        return dist_prob

def train(svi, train_loader,adj, use_cuda=True):
    epoch_loss = 0
    for x in train_loader:
        if use_cuda:
            x = x.cuda()
            adj = adj.cuda()
        epoch_loss += svi.step(x,adj)

    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

celldata = H5ADataSet("preprocessed.h5ad")
adata = anndata.read_h5ad("preprocessed.h5ad")

NUMGENS = len(adata.var)
dimZ = 64
dimG = 128
BATCHSIZE = 7284
LEARN_RATE = 1.0e-4
USE_CUDA = True
NUM_EPOCHS = 30
nclusters = 20
cell_loader = DataLoader(celldata, batch_size=BATCHSIZE)
#kmeans = KMeans(n_clusters= 13).fit(celldata.data.X)
#ARI = adjusted_rand_score(adata.obs['transf_annotation'],kmeans.labels_)
#NMI = normalized_mutual_info_score(adata.obs['transf_annotation'],kmeans.labels_)
#print("ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))

pyro.clear_param_store()

vae = VAE(NUMGENS, dimZ, dimG,nclusters, 0.5)
vae.cuda()
adam_args ={"lr":LEARN_RATE}
optimizer = ClippedAdam(adam_args)

elbo = Trace_ELBO(strict_enumeration_warning=False)
svi = SVI(vae.model, vae.guid, optimizer, loss=elbo)

gc.collect()
torch.cuda.empty_cache()

train_elbo = []
test_elbo = []

adj = np.loadtxt('mix105_graph.csv',delimiter=',')
adj = np.array(adj)
adj = adj.tolist()
adj = torch.FloatTensor(adj)


for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, cell_loader,adj, use_cuda=USE_CUDA)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d] average training loss: %.4f" % (epoch, total_epoch_loss_train))

TZ = []
for x in cell_loader:
    x = x.cuda()
    adj = adj.cuda()
    z = vae.getZ(x,adj)
    zz = z.cpu().detach().numpy().tolist()
    TZ+=zz
TZ = np.array(TZ)
TZ = torch.tensor(TZ)

reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
zr = reducer.fit_transform(TZ)
adata.obsm["umap"] = zr
#print(adata.obs)
scanpy.pl.umap(adata,color='leiden', palette=scanpy.pl.palettes.default_20,  legend_loc = 'on data',legend_fontsize = 12,legend_fontoutline=2)
kmeans = KMeans(n_clusters = 20).fit(TZ)
ARI = adjusted_rand_score(adata.obs['leiden'], kmeans.labels_)
NMI = normalized_mutual_info_score(adata.obs['leiden'],kmeans.labels_)
print("ARI: "+str(ARI)+"  "+"NMI: "+str(NMI))
