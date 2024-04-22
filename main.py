import openpyxl
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split,Subset
from transformers import BertConfig, BertTokenizer, BertModel
import transformers

import Configs
import models.overall_model
import Dataset
##


configs=Configs.Configs()
dataset_build=Dataset.Dataset_build(configs)

contents_train,contents_test=dataset_build.build_dataset_with_txt()

train_iter=Dataset.Dataset_iterater(contents_train,configs)
test_iter=Dataset.Dataset_iterater(contents_test,configs)


print("+----------------------------------------------------------------------------------------------------------------------------------------------------------------+")

net=models.overall_model.Net(configs).to(configs.device)
optimizer = torch.optim.Adam(net.parameters(), lr=configs.learning_rate)



print("----------------------training start----------------------")
def f_train():
    net.train()
    for i,(datas,labels) in enumerate(train_iter):

        net.zero_grad()
        num_b = datas[0].size(0)
        h0=net.my_bi_gru.inithidden(num_b)
        outputs=net(datas,h0,num_b)
        p1,p2,p3,p4=outputs

        if configs.soft_target==False:
            y1 = torch.LongTensor([i[0] for i in labels]).to(configs.device)
            y2 = torch.LongTensor([i[1] for i in labels]).to(configs.device)
            y3 = torch.LongTensor([i[2] for i in labels]).to(configs.device)
            y4 = torch.LongTensor([i[3] for i in labels]).to(configs.device)

            # y1 =[label[0] for label in labels]
            # y2 =[label[1] for label in labels]
            # y3 =[label[2] for label in labels]
            # y4 =[label[3] for label in labels]
            loss1 = F.cross_entropy(p1, y1)
            loss2 = F.cross_entropy(p2, y2)
            loss3 = F.cross_entropy(p3, y3)
            loss4 = F.cross_entropy(p4, y4)
            pass
        else :

            y1i = torch.LongTensor([i[0] for i in labels]).to(configs.device)
            y2i = torch.LongTensor([i[1] for i in labels]).to(configs.device)
            y3i = torch.LongTensor([i[2] for i in labels]).to(configs.device)
            y4i = torch.LongTensor([i[3] for i in labels]).to(configs.device)

            # y1 =[label[0] for label in labels]
            # y2 =[label[1] for label in labels]
            # y3 =[label[2] for label in labels]
            # y4 =[label[3] for label in labels]
            loss1i = F.cross_entropy(p1, y1i)
            loss2i = F.cross_entropy(p2, y2i)
            loss3i = F.cross_entropy(p3, y3i)
            loss4i = F.cross_entropy(p4, y4i)



            y1 = [i[0] for i in labels]
            y2 = [i[1] for i in labels]
            y3 = [i[2] for i in labels]
            y4 = [i[3] for i in labels]


            y1 = torch.tensor([configs.soft_100[i] for i in y1]).to(configs.device)
            y2 = torch.tensor([configs.soft_10[i] for i in y2]).to(configs.device)
            y3 = torch.tensor([configs.soft_10[i] for i in y3]).to(configs.device)
            y4 = torch.tensor([configs.soft_10[i] for i in y4]).to(configs.device)

            y1 = F.softmax(y1, dim=1)
            y2 = F.softmax(y2, dim=1)
            y3 = F.softmax(y3, dim=1)
            y4 = F.softmax(y4, dim=1)

            # loss1=F.cross_entropy(p1,y1)
            # loss2=F.cross_entropy(p2,y2)
            # loss3=F.cross_entropy(p3,y3)
            # loss4=F.cross_entropy(p4,y4)
            ##x_log = F.log_softmax(x, dim=1)

            p1 = F.log_softmax(p1, dim=1)
            p2 = F.log_softmax(p2, dim=1)
            p3 = F.log_softmax(p3, dim=1)
            p4 = F.log_softmax(p4, dim=1)

            # p1=torch.log(p1)
            # p2=torch.log(p2)
            # p3 = torch.log(p3)
            # p4 = torch.log(p4)
            # # print(p2)
            #
            # # print(y2[0])
            # # print("[]    []    []")
            #
            #
            # loss1=torch.sum(p1*y1)
            # loss2=torch.sum(p2*y2)
            # loss3=torch.sum(p3*y3)
            # loss4=torch.sum(p4*y4)
            criterion = torch.nn.KLDivLoss().to(configs.device)

            loss1=criterion(p1,y1)
            loss2=criterion(p2,y2)
            loss3=criterion(p3,y3)
            loss4=criterion(p4,y4)

            loss1=(configs.eps)*loss1+(1-configs.eps)*loss1i
            loss2=(configs.eps)*loss2+(1-configs.eps)*loss2i
            loss3=(configs.eps)*loss3+(1-configs.eps)*loss3i
            loss4=(configs.eps)*loss4+(1-configs.eps)*loss4i


            pass


        loss=loss1+loss2+loss3+loss4

        loss.backward()
        optimizer.step()
        if i%200==0:
            print("i = ",i," [] loss = ",loss)
            print("loss : ",(loss1).item()," ",loss2.item()," ",loss3.item()," ",loss4.item())
            pass
        pass
    pass

def f_test():
    net.eval()
    correct=torch.Tensor([0.0,0.0,0.0,0.0])
    total=torch.Tensor([0.0])
    correct=correct.to(configs.device)
    total=total.to(configs.device)
    with torch.no_grad():
        for i,(datas,labels) in enumerate(test_iter):
            net.zero_grad()
            num_b = datas[0].size(0)
            h0 = net.my_bi_gru.inithidden(num_b)
            outputs = net(datas, h0, num_b)

            y1 = [i[0] for i in labels]
            y2 = [i[1] for i in labels]
            y3 = [i[2] for i in labels]
            y4 = [i[3] for i in labels]
            p1, p2, p3, p4 = outputs

            p1 = torch.max(p1, 1)[1]
            p2 = torch.max(p2, 1)[1]
            p3 = torch.max(p3, 1)[1]
            p4 = torch.max(p4, 1)[1]

            total[0]+=num_b

            for i in range(num_b):
                correct[0] += configs.score_100[y1[i]][p1[i].item()]
                correct[1] += configs.score_10[y2[i]][p2[i].item()]
                correct[2] += configs.score_10[y3[i]][p3[i].item()]
                correct[3] += configs.score_10[y4[i]][p4[i].item()]
                pass

            #only output the prediction value of first example in each batch


            if i % 400 == 0:
                print("label : ", y1[0].item(), " ", y2[0].item(), " ", y3[0].item(), " ", y4[0].item())
                print("predict : ", (p1[0]).item(), " ", p2[0].item(), " ", p3[0].item(), " ", p4[0].item())
                pass

            pass
        pass
    print("-------result---------")
    print(correct)
    print(total)

    L1=100*correct[0]/total[0]
    L2=100*correct[1]/total[0]
    L3=100*correct[2]/total[0]
    L4=100*correct[3]/total[0]



    print(' L_possibility accuracy            : {}'.format(100*correct[0]/total[0]))
    print(' L_seriousness accuracy            : {}'.format(100*correct[1]/total[0]))
    print(' L_Risk_impact_factors accuracy    : {}'.format(100*correct[2]/total[0]))
    print(' L_Risk_diffusion_degree accuracy  : {}'.format(100*correct[3]/total[0]))

    avg=(L1+L2+L3+L4)/4.0
    print("AVG  : ",avg)

    pass

print('+-------------------------------------------------------------------------------------------------------------+[start]')

for i in range(configs.epoch):
    print("epoch  ----------------------------------------------- : ",i)
    f_train()
    f_test()
    pass


print("+------------------------------------------------------------------------------------------------------------------+[end]")




