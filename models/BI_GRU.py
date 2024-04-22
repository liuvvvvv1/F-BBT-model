import torch
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer, BertModel
import transformers
import copy
import math

dropout=0.1

class BI_GRU_Net(torch.nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.emb=torch.nn.Embedding(configs.vac_size,configs.word_size//2)
        self.my_gru=torch.nn.GRU(input_size=configs.word_size//2,hidden_size=configs.hidden_size//2,num_layers=1,bidirectional=True,batch_first=True)
        self.hidden_size=configs.hidden_size//2
        self.max_len=configs.max_len
        self.device=configs.device
        self.dropout=torch.nn.Dropout(0.1)
        #---------------------------------------------------------------------------------------------------------------
        # c=copy.deepcopy
        # attn=Mul_attention(2,configs.max_len,0.1)
        # ff=FeedForward(configs.max_len,configs.max_len*2)
        # self.encoder_forward=Encoder(Encoder_layer(configs.max_len,c(attn),c(ff),0.1),6)
        # self.encoder_backward=Encoder(Encoder_layer(configs.max_len,c(attn),c(ff),0.1),6)

        self.bi=configs.Bi


        pass

    # def init(self):
    #     for p in self.my_gru.parameters():
    #         if p.dim() > 1:
    #             torch.nn.init.xavier_uniform_(p)
    #             pass
    #         pass
    #
    #     for p in self.encoder_forward.parameters():
    #         if p.dim() > 1:
    #             torch.nn.init.kaiming_normal_(p)
    #             pass
    #         pass
    #
    #     for p in self.encoder_backward.parameters():
    #         if p.dim() > 1:
    #             torch.nn.init.kaiming_normal_(p)
    #             pass
    #         pass
    #
    #     pass

    def forward(self,x,hidden,num_b):
        if self.bi==0:
            num_b = x.size(0)
            return torch.zeros(num_b, self.max_len,self.hidden_size*2).to(self.device),torch.zeros(num_b, self.hidden_size*2).to(self.device)
            pass
        #self.init()
        emb=self.emb(x)
        emb=self.dropout(emb)
        out,h=self.my_gru(emb,hidden)
        out = out.reshape((num_b,self.max_len,2,self.hidden_size))


        # output_forward = out[:, self.max_len - 1, 0, :].squeeze()
        # output_backward = out[:, 0, 1, :].squeeze()

        output_forward = out[:,:, 0, :].squeeze()
        output_backward = out[:,:, 1, :].squeeze()

        x = [output_forward, output_backward]
        x = torch.cat(x, dim=-1)

        #print("---------------------------------------------------------")
        output_forwardi = out[:, self.max_len - 1, 0, :].squeeze()
        output_backwardi = out[:, 0, 1, :].squeeze()
        xi = [output_forwardi, output_backwardi]
        xi = torch.cat(xi, dim=1)


        return x,xi


        # else:
        #
        #     output_forwardi = out[:, self.max_len - 1, 0, :].squeeze()
        #     output_backwardi = out[:, 0, 1, :].squeeze()
        #     xi = [output_forwardi, output_backwardi]
        #     xi = torch.cat(xi, dim=1)
        #
        #
        #     output_forward = out[:, :, 0, :].squeeze()
        #     output_backward = out[:, :, 1, :].squeeze()
        #     # --------------------------------------------------
        #     # x=[output_forward,output_backward]
        #     # x=torch.cat(x,dim=1)
        #
        #     output_forward = output_forward.transpose(-1, -2)
        #     output_backward = output_backward.transpose(-1, -2)
        #
        #     output_forward = self.encoder_forward(output_forward)
        #     output_backward = self.encoder_backward(output_backward)
        #
        #     # output_forward = torch.mean(output_forward, dim=-1)####
        #     # output_backward = torch.mean(output_backward, dim=-1)####
        #
        #     output_forward = output_forward[:,:,0].squeeze()
        #     output_backward = output_backward[:,:,0].squeeze()
        #
        #     ans = torch.cat((output_forward, output_backward), dim=1)
        #
        #     return ans+xi
        #     pass
        pass
    def inithidden(self,num_b):
        return torch.zeros(2,num_b,self.hidden_size).to(self.device)
        pass

    pass

print("import bi_gru")

# print("-----------------------------------------------------------------this is a attention mechanism-----------------------------------------------------------------")
# print("----------------------------------------------------------------- applied on BI_GRU_Net -----------------------------------------------------------------")
#
# def attention(q,k,v,dropout=None):
#     dk=q.size(-1)
#     score=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dk)
#     attn=F.softmax(score,dim=-1)
#     if dropout is not None:
#         attn=dropout(score)
#         pass
#     ans=torch.matmul(attn,v)
#     return ans,attn
#     pass
#
# def clones(mod,n):
#     return torch.nn.ModuleList([copy.deepcopy(mod) for i in range(n)])
#     pass
#
#
# class Mul_attention(torch.nn.Module):
#     def __init__(self,head,word_size,dropout):
#         super().__init__()
#         assert word_size%head==0
#         self.dk=word_size//head
#         self.head=head
#         self.word_size=word_size
#         self.linears=clones(torch.nn.Linear(word_size,word_size),3)
#         self.linear_last=torch.nn.Linear(word_size,word_size)
#         self.attn=None
#         self.dropout=torch.nn.Dropout(dropout)
#         pass
#     def forward(self,q,k,v):
#         batch_size=q.size(0)
#         q,k,v=[lin(x).view(batch_size,-1,self.head,self.dk).transpose(1,2) for lin,x in zip(self.linears,(q,k,v))]
#         x,self.attn=attention(q,k,v,dropout=self.dropout)
#         x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.dk)
#         return self.linear_last(x)
#         pass
#     pass
#
#
# class FeedForward(torch.nn.Module):
#     def __init__(self,word_size,d_ff,dropout=0.1):
#         super().__init__()
#         self.w1=torch.nn.Linear(word_size,d_ff)
#         self.w2=torch.nn.Linear(d_ff,word_size)
#         self.dropout=torch.nn.Dropout(dropout)
#         pass
#     def forward(self,x):
#         return self.w2(self.dropout(F.relu(self.w1(x))))
#         pass
#     pass
#
#
# class Layernorm(torch.nn.Module):
#     def __init__(self,features,eps=1e-6):
#         super().__init__()
#         self.a2=torch.nn.Parameter(torch.ones(features))
#         self.b2=torch.nn.Parameter(torch.zeros(features))
#         self.eps=eps
#         pass
#     def forward(self,x):
#         mean=x.mean(-1,keepdim=True)
#         std=x.std(-1,keepdim=True)
#         return self.a2*(x-mean)/(std+self.eps)+self.b2
#         pass
#     pass
#
# print("---------------------------------------------------------------------------------------------------------------")
#
# class Sublayer_connection(torch.nn.Module):
#     def __init__(self,size,dropout=0.1):
#         super().__init__()
#         self.norm=Layernorm(size)
#         self.dropout=torch.nn.Dropout(dropout)
#         self.size=size
#         pass
#     def forward(self,x,sublayer):
#         return x+self.dropout(sublayer(self.norm(x)))
#         pass
#     pass
#
# class Encoder_layer(torch.nn.Module):
#     def __init__(self,size,self_attn,feed_forward,dropout):
#         super().__init__()
#         self.self_attn=self_attn
#         self.feed_forward=feed_forward
#         self.size=size
#         self.sublayer=clones(Sublayer_connection(size,dropout),2)
#         pass
#     def forward(self,x):
#         x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x))
#         x=self.sublayer[1](x,self.feed_forward)
#         return x
#     pass
#
#
#
# class Encoder(torch.nn.Module):
#     def __init__(self,layer,n):
#         super().__init__()
#         self.layers=clones(layer,n)
#         self.norm=Layernorm(layer.size)
#         pass
#     def forward(self,x):
#         for layer in self.layers:
#             x=layer(x)
#             pass
#         return self.norm(x)
#         pass
#     pass
#

