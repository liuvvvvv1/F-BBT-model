import copy
import math
import torch.nn.functional as F
import torch
from models import BERT, BI_GRU, TEXT_CNNS


class Net(torch.nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.my_bert= BERT.BERT_Net(configs)
        self.my_text_cnns= TEXT_CNNS.TEXTCNNS_Net(configs)
        self.my_bi_gru= BI_GRU.BI_GRU_Net(configs)

        self.max_len=configs.max_len
        self.hidden_size=configs.hidden_size
        self.device=configs.device

        self.bi=configs.Bi
        self.tc=configs.Tc
        self.opt=configs.opt
        self.num_layer=configs.num_layer

        self.L_possibility=torch.nn.Linear(configs.hidden_size*3,100)
        self.L_seriousness=torch.nn.Linear(configs.hidden_size*3,configs.num_class)
        self.L_Risk_impact_factors=torch.nn.Linear(configs.hidden_size*3,configs.num_class)
        self.L_Risk_diffusion_degree=torch.nn.Linear(configs.hidden_size*3,configs.num_class)
        self.L_link=torch.nn.Linear(configs.hidden_size,configs.hidden_size*3)


        c=copy.deepcopy
        attn=Mul_attention(4,configs.hidden_size,0.1)
        ff=FeedForward(configs.hidden_size,configs.hidden_size*2)
        self.encoder=Encoder(Encoder_layer(configs.hidden_size,c(attn),c(ff),0.1),self.num_layer)

        pass

    def forward(self,x,hidden,num_b):
        # mask1=x[2]
        # mask2=torch.ones((num_b,self.max_len)).to(self.device)
        # mask3=x[2]
        # mask=torch.cat([mask1,mask2,mask3],dim=1).unsqueeze(-1)


        #print("mask")
        # print(mask)

        # print("[]")
        #print(mask.shape)
        # print("[][][][][][]")

        global d
        out1,out1i=self.my_bert(x)
        out2,out2i=self.my_text_cnns(x[0])
        out3,out3i=self.my_bi_gru(x[0],hidden,num_b)



        if self.opt==0:
            # if self.bi==0:
            #     out3=out3*0
            #     pass
            # if self.tc==0:
            #     out2=out2*0
            #     pass

            # out = [out1i, out2i, out3i]
            # out = torch.cat(out, dim=1)

            out=out1i+out2i+out3i

            possibility = self.L_possibility(out)
            seriousness = self.L_seriousness(out)
            Risk_impact_factors = self.L_Risk_impact_factors(out)
            L_Risk_diffusion_degree = self.L_Risk_diffusion_degree(out)
            # print("[]")
            # print(possibility.shape)
            return (possibility, seriousness, Risk_impact_factors, L_Risk_diffusion_degree)
            pass
        elif self.opt==1:
            emb1 = torch.ones((num_b, self.max_len, self.hidden_size)).to(self.device)
            emb2 = (2 * torch.ones((num_b, self.max_len, self.hidden_size))).to(self.device)
            emb3 = (3 * torch.ones((num_b, self.max_len, self.hidden_size))).to(self.device)

            out1 = out1 + emb1
            out2 = out2 + emb2
            out3 = out3 + emb3
            d = torch.cat([out1, out2, out3], dim=-2)
            d = self.encoder(d)
            d = d[:, 0, :].squeeze()
            d = self.L_link(d)


            possibility = self.L_possibility(d)
            seriousness = self.L_seriousness(d)
            Risk_impact_factors = self.L_Risk_impact_factors(d)
            L_Risk_diffusion_degree = self.L_Risk_diffusion_degree(d)

            return (possibility, seriousness, Risk_impact_factors, L_Risk_diffusion_degree)

            pass
        else:
            emb1=torch.ones((num_b,self.max_len,self.hidden_size)).to(self.device)
            emb2=(2*torch.ones((num_b,self.max_len,self.hidden_size))).to(self.device)
            emb3=(3*torch.ones((num_b,self.max_len,self.hidden_size))).to(self.device)

            out1=out1+emb1
            out2=out2+emb2
            out3=out3+emb3
            d = torch.cat([out1,out2, out3], dim=-2)
           # print(d.shape)
            d=self.encoder(d)

            d=d[:,0,:].squeeze()
            #out1i=out1i+d
            #print(d.shape)
            d=self.L_link(d)

            # outi = [out1i, out2i, out3i]
            # outi = torch.cat(outi, dim=1)
            # out=out+outi
            #out=torch.cat([out1i,d],dim=1)

            outi = [out1i, out2i*0, out3i*0]
            outi = torch.cat(outi, dim=1)

            possibility = self.L_possibility(outi)
            seriousness = self.L_seriousness(d)
            Risk_impact_factors = self.L_Risk_impact_factors(d)
            L_Risk_diffusion_degree = self.L_Risk_diffusion_degree(d)
            # print("[]")
            # print(possibility.shape)
            return (possibility, seriousness, Risk_impact_factors, L_Risk_diffusion_degree)

            pass

        pass
    pass







print("-----------------------------------------------------------------this is a attention mechanism-----------------------------------------------------------------")

def attention(q,k,v,mask=None,dropout=None):
    dk=q.size(-1)
    score=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dk)
    if mask is not None:#########
        score=score.masked_fill(mask==0,-1e9)
        pass
    p_attn=F.softmax(score,dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)
        pass
    b=torch.matmul(p_attn,v)
    return b,p_attn
    pass

def clones(mod,n):
    return torch.nn.ModuleList([copy.deepcopy(mod) for i in range(n)])
    pass


class Mul_attention(torch.nn.Module):
    def __init__(self,head,word_size,dropout):
        super().__init__()
        assert word_size%head==0
        self.dk=word_size//head
        self.head=head
        self.word_size=word_size
        self.linears=clones(torch.nn.Linear(word_size,word_size),3)
        self.linear_last=torch.nn.Linear(word_size,word_size)
        self.attn=None
        self.dropout=torch.nn.Dropout(dropout)
        pass
    def forward(self,q,k,v,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)
            pass
        batch_size=q.size(0)
        q,k,v=[lin(x).view(batch_size,-1,self.head,self.dk).transpose(1,2) for lin,x in zip(self.linears,(q,k,v))]
        x,self.attn=attention(q,k,v,mask=mask,dropout=self.dropout)
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.dk)
        return self.linear_last(x)
        pass
    pass


class FeedForward(torch.nn.Module):
    def __init__(self,word_size,d_ff,dropout=0.1):
        super().__init__()
        self.w1=torch.nn.Linear(word_size,d_ff)
        self.w2=torch.nn.Linear(d_ff,word_size)
        self.dropout=torch.nn.Dropout(dropout)
        pass
    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
        pass
    pass


class Layernorm(torch.nn.Module):
    def __init__(self,features,eps=1e-6):
        super().__init__()
        self.a2=torch.nn.Parameter(torch.ones(features))
        self.b2=torch.nn.Parameter(torch.zeros(features))
        self.eps=eps
        pass
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a2*(x-mean)/(std+self.eps)+self.b2
        pass
    pass

print("---------------------------------------------------------------------------------------------------------------")

class Sublayer_connection(torch.nn.Module):
    def __init__(self,size,dropout=0.1):
        super().__init__()
        self.norm=Layernorm(size)
        self.dropout=torch.nn.Dropout(dropout)
        self.size=size
        pass
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))
        pass
    pass

class Encoder_layer(torch.nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super().__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.size=size
        self.sublayer=clones(Sublayer_connection(size,dropout),2)
        pass
    def forward(self,x,mask=None):
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        x=self.sublayer[1](x,self.feed_forward)
        return x
    pass



class Encoder(torch.nn.Module):
    def __init__(self,layer,n):
        super().__init__()
        self.layers=clones(layer,n)
        self.norm=Layernorm(layer.size)
        pass
    def forward(self,x,mask=None):
        for layer in self.layers:
            x=layer(x,mask)
            pass
        return self.norm(x)
        pass
    pass




