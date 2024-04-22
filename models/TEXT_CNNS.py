import torch
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import random_split,Subset
from transformers import BertConfig, BertTokenizer, BertModel
import transformers

dropout=0.1
num_conv=4

class TEXTCNNS_Net(torch.nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.out_c=configs.max_len

        self.emb=torch.nn.Embedding(configs.vac_size,configs.word_size)
        self.conv_list=torch.nn.ModuleList()
        for i in range(1,num_conv+1):
            self.conv_list.append(torch.nn.Conv2d(in_channels=1,out_channels=self.out_c,kernel_size=(i,configs.word_size)))
            pass
        pass
        self.lt=torch.nn.Linear(self.out_c*num_conv,configs.hidden_size)

        self.all_lin=torch.nn.Linear(num_conv,configs.hidden_size)


        self.dropout=torch.nn.Dropout(dropout)
        self.tc=configs.Tc
        self.hidden_size=configs.hidden_size
        self.device=configs.device
        self.max_len=configs.max_len

    def forward(self,x):

        if self.tc==0:
            num_b=x.size(0)
            return torch.zeros(num_b, self.max_len,self.hidden_size).to(self.device),torch.zeros(num_b, self.hidden_size).to(self.device)
            pass

        x=x.long()
        out=self.dropout(self.emb(x))
        out=out.unsqueeze(1)              # conv2d[b,(c),h,w]
        conved=[F.relu(conv(out)).squeeze(3) for conv in self.conv_list]


        # print("con")
        # print(len(conved))
        # print(conved[0].shape)
        # print(conved[1].shape)
        # print(conved[2].shape)
        # print(conved[3].shape)
        # torch.Size([16, 64, 64])
        # torch.Size([16, 64, 63])
        # torch.Size([16, 64, 62])
        # torch.Size([16, 64, 61])



        # # --
        # if self.tc==2:
        #     conveds = [torch.mean(conv, dim=-1).squeeze() for conv in conved]
        #     conveds = [self.lin1(conv) for conv in conveds]
        #     conveds = [F.relu(conv) for conv in conveds]
        #     conveds = [self.lin2(conv) for conv in conveds]
        #     conveds = [F.sigmoid(conv) for conv in conveds]
        #     conveds = [conv.unsqueeze(-1) for conv in conveds]
        #
        #     #conved = [conv * convs for conv, convs in zip(conved, conveds)]
        #     convedss = [conv * convs for conv, convs in zip(conved, conveds)]
        #     conved=[conv + convs for conv, convs in zip(conved, convedss)]
        #     pass
        # #--

        pooled=[F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]

        # print("pooled")
        # print(len(pooled))
        # print(pooled[0].shape)torch.Size([16, 8])
        # print(pooled[1].shape)torch.Size([16, 8])
        # print(pooled[2].shape)torch.Size([16, 8])
        # print(pooled[3].shape)torch.Size([16, 8])


        cat=torch.cat(pooled,dim=1)
        outi=F.relu(self.lt(cat))
        #print(outi.shape)
        ##############################################################################################################
        conveds=[F.max_pool1d(conv,conv.shape[2]) for conv in conved]
        # print("coned")
        # print(len(conveds))
        # print(conveds[0].shape)
        # print(conveds[1].shape)
        # print(conveds[2].shape)
        # print(conveds[3].shape)

        out=torch.cat(conveds,dim=-1)
        #print("conveds",out.shape)
        out=self.all_lin(out)
        # print("out")
        # print(out.shape)
        return out,outi
        pass
    pass





print("import text_cnn")