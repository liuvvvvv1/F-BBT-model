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



#path_bert="../bert_model"
#tokenizer = BertTokenizer.from_pretrained(path_bert)
#BERT_model = BertModel.from_pretrained(path_bert)
print("+----------------------------------------------------------------------------------------------------------------------------------------------------------------+")

#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BERT_Net(torch.nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.bert=configs.BERT_model
        self.hidden_size=configs.hidden_size
        self.Be=configs.Be
        self.max_len=configs.max_len
        self.device=configs.device
        for p in self.bert.parameters():
            p.requires_grad=True
            pass

        pass
    def forward(self,x):
        context=x[0]
        if self.Be==0:
            num_b=context.size(0)
            return torch.zeros(num_b, self.max_len,self.hidden_size).to(self.device),torch.zeros(num_b, self.hidden_size).to(self.device)
            pass

        mask=x[2]
        #_,y=self.bert(context,attention_mask=mask,return_dict = False)
        y, yi = self.bert(context, attention_mask=mask, return_dict=False)

        #print(ah.shape)

        return y,yi
        pass
    pass

print("import bert")




