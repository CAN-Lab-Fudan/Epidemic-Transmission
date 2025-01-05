import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, BatchNorm, global_max_pool, LayerNorm
from typing import Union, Any


class PreStream(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super(PreStream, self).__init__()

        self.f_in = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = LayerNorm(in_channels=out_features)
        # self.norm = BatchNorm(in_channels=out_features)
        self.dropout = nn.Dropout()

    def forward(self, data: Data):
        x = self.f_in.forward(input=data.x)
        x = self.norm.forward(x=x, batch=data.batch)
        # x = self.norm.forward(x=x)
        x = F.relu(input=x)
        x = self.dropout.forward(input=x)


        return x


class My_GATConv(nn.Module):

    def __init__(self, in_features: int, hidden_features_a: int, out_features: int, dropout: float=0.1) -> None:
        super(My_GATConv, self).__init__()
        self.f_att_1 = GATConv(in_channels=in_features, out_channels=hidden_features_a, heads=8, concat=False, dropout=dropout)
        self.norm1 = LayerNorm(in_channels=hidden_features_a)

        self.f_att_2 = GATConv(in_channels=hidden_features_a, out_channels=out_features, heads=8, concat=False, dropout=dropout)
        self.norm2 = LayerNorm(in_channels=out_features)

        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, x: Union[torch.Tensor, SparseTensor], data: Data):
        # 加入残差连接，使得训练更不容易过拟合，在数据较为简单的情况下使用
        x =x + self.f_att_1.forward(x=x, edge_index=data.edge_index)
        x = self.norm1.forward(x=x, batch=data.batch)
        x = F.relu(input=x)
        x = self.drop_out.forward(input=x)

        x = x + self.f_att_2.forward(x=x, edge_index=data.edge_index)
        x = self.norm2.forward(x=x, batch=data.batch)
        x = F.relu(input=x)
        x = self.drop_out.forward(input=x)

        return x


class DownStream(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(DownStream, self).__init__()

        self.f_downstream = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor):
        x = self.f_downstream.forward(input=x)

        return x
    

class DynamicGAT(nn.Module):
    def __init__(self, args: Any) -> None:
        super(DynamicGAT, self).__init__()
        self.pre_linear = PreStream(in_features=args.in_features, out_features=args.f_att_in)
        # self.f_att = My_GATConv(in_features=args.f_att_in, hidden_features_a=args.f_att_hidden, out_features=args.f_out_in)
        self.f_att = My_GATConv(in_features=args.in_features, hidden_features_a=args.f_att_hidden, out_features=args.f_out_in)
        self.f_linear = DownStream(in_features=args.f_out_in, out_features=args.out_features)

    def forward(self, data: Data):
        # x = self.pre_linear.forward(data=data)
        # x = self.f_att.forward(x=x, data=data)
        x = self.f_att.forward(x=data.x, data=data)
        x = global_mean_pool(x=x, batch=data.batch)
        x = self.f_linear.forward(x=x)

        return x
