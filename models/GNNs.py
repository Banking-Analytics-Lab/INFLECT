from re import search
import torch.nn as nn 
import torch 
from torch_geometric.nn import GCNConv, GATv2Conv,GINEConv,GraphSAGE
from torch.nn import Linear, LayerNorm
import torch.nn.functional as F



class GNN(nn.Module):
    def __init__(self,input_dim=None, embedding_dim=None,output_dim=None,edge_dim = None,heads = 4, n_layers = 5) -> None:
        super().__init__()
    def forward(self):
        pass 


class GATs(nn.Module): 
    def __init__(self,input_dim, embedding_dim,output_dim,edge_dim,heads, n_layers, dropout_rate, **kwargs) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.gat1 = GATv2Conv(input_dim, embedding_dim, heads=heads,edge_dim = edge_dim) # dim_h * num heads
        self.layer_norm1 = LayerNorm(embedding_dim*heads, elementwise_affine=True)
        self.GAT_list = torch.nn.ModuleList([GATv2Conv(embedding_dim*heads, embedding_dim, heads=heads,edge_dim = edge_dim)  for _ in range(n_layers-2)])
        self.gat2 = GATv2Conv(embedding_dim*heads, output_dim, heads=1, edge_dim = edge_dim)
        self.layer_norm2 = LayerNorm(output_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inp, edge_index,edge_feats):

        h = self.gat1(inp, edge_index,edge_attr = edge_feats)
        h = F.elu(h)
        h = self.dropout(h)
        for l in self.GAT_list:
            h = l(h, edge_index,edge_attr = edge_feats)
            h = self.layer_norm1(h)
            h = F.elu(h)
            h = self.dropout(h)
        h = self.gat2(h, edge_index, edge_feats)
        h = self.layer_norm2(h)
        h = F.elu(h)
        h = self.dropout(h)
        return h
    

class GCNs(GNN):
  """Graph Convolutional Network"""
  def __init__(self, input_dim,embedding_dim, output_dim,n_layers,dropout_rate, **kwargs):
    super().__init__()
    self.gcn1 = GCNConv(input_dim, embedding_dim)
    self.layer_norm1 = LayerNorm(embedding_dim, elementwise_affine=True)
    self.layer_norm2 = LayerNorm(output_dim, elementwise_affine=True)
    self.dropout_rate = dropout_rate
    self.GCN_list = torch.nn.ModuleList([GCNConv(embedding_dim, embedding_dim)  for _ in range(n_layers-1)])
    self.lin = Linear(embedding_dim, output_dim)
    self.dropout = nn.Dropout(self.dropout_rate)

  def forward(self, x, edge_index,edge_feats):
    h = self.gcn1(x, edge_index,edge_weight = edge_feats)
    h = self.layer_norm1(h)
    h = F.elu(h)
    h = self.dropout(h)
    for (i,l) in enumerate(self.GCN_list):
        if i == len(self.GCN_list) - 1: 
            h = l(h, edge_index,edge_weight = edge_feats)
            h = self.layer_norm2(h)
            h = F.elu(h)
            h = self.dropout(h)
        else: 
            h = l(h, edge_index,edge_weight = edge_feats)
            h = self.layer_norm1(h)
            h = F.elu(h)
            h = self.dropout(h)
    h = self.lin(h)
    h = F.relu(h)
    return h


class GINs(GNN):
    def __init__(self,input_dim, embedding_dim,output_dim,edge_dim,dropout_rate, n_layers,train_eps = False,eps=0,**kwargs) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        h_theta_0 = [nn.Sequential(nn.Linear(input_dim,embedding_dim))]
        lin_list = [nn.Sequential(nn.Linear(embedding_dim, embedding_dim))  for _ in range(n_layers-2)]
        h_theta_n = [nn.Sequential(nn.Linear(embedding_dim,output_dim ))]
        self.lin_list = nn.ModuleList(h_theta_0 + lin_list+h_theta_n)
        self.GIN_list = nn.ModuleList()
        self.layer_norm1 = LayerNorm(embedding_dim, elementwise_affine=True)
        self.layer_norm2 = LayerNorm(output_dim, elementwise_affine=True)
        for h_theta_i in self.lin_list:
            self.GIN_list.append(GINEConv(h_theta_i,eps,train_eps=train_eps,edge_dim=edge_dim))
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self,x ,edge_index,edge_feats):
        h = x

        for (i,l) in enumerate(self.GIN_list):
            if i == len(self.GIN_list) - 1: 
                h = l(h,edge_index,edge_feats)
                h = self.layer_norm2(h)
                h = F.elu(h)
                h = self.dropout(h)
            else: 
                h = l(h,edge_index,edge_feats)
                h = self.layer_norm1(h)
                h = F.elu(h)
                h = self.dropout(h)

        return h 


class SAGEs(GNN):
    def __init__(self, input_dim ,embedding_dim, output_dim, search_depth,n_layers,dropout_rate,**kwargs) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate 
        l1 = GraphSAGE(input_dim,embedding_dim,search_depth,output_dim,dropout=dropout_rate) 
        self.SAGE_list = nn.ModuleList([GraphSAGE(output_dim,embedding_dim,search_depth,output_dim,dropout_rate) for _ in range(n_layers - 1) ])
        self.SAGE_list.insert(0,l1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self,x,edge_index,edge_feats):
        h= x 
        for l in self.SAGE_list:
            h  = l(h,edge_index)
            h = F.elu(h)
            h = self.dropout(h)
        return h 
    

def get_GNN(gnn ):
    
    if gnn == 'GAT':
        return GATs
    elif gnn == 'GIN':
        return GINs
    elif gnn == 'SAGE':
        return SAGEs
    else: 
        return GCNs