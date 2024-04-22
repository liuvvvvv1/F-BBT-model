import copy

import torch
from transformers import BertConfig, BertTokenizer, BertModel



class Configs:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 0.00001
        self.epoch = 5
        self.batch_size = 16
        self.max_len = 32
        self.vac_size = 21128
        self.hidden_size= 768
        self.word_size= 768
        self.num_class = 10
        self.path_bert = "./bert_model"
        self.path_text = './datas/text.txt'
        self.tokenizer = BertTokenizer.from_pretrained(self.path_bert)
        self.BERT_model=BertModel.from_pretrained(self.path_bert)

        self.Bi=1
        self.Tc=1
        self.Be=1
        self.opt=1
        self.soft_target=False
        self.num_layer=2
        self.eps=0.05

        self.score_10=[]
        self.soft_10=[]
        self.score_100=[]
        self.soft_100=[]
        c = copy.deepcopy

        for i in range(0,10):
            id=i
            t=[]
            for j in range(0,10):
                it=j
                if it==id:
                    t.append(1.0)
                elif abs(it-id)==1:
                    t.append(0.7)
                elif abs(it-id)==2:
                    t.append(0.3)
                else:
                    t.append(0.0)
                pass
            self.score_10.append(t)
            pass

            self.soft_10=c(self.score_10)

        # for i in range(0,10):
        #     t=[]
        #     sum=0.0
        #     for j in range(0,10):
        #         sum+=self.score_10[i][j]
        #         pass
        #     for j in range(0,10):
        #         t.append(self.score_10[i][j]/sum)
        #         pass
        #     self.soft_10.append(t)
        #     pass

        # for i in range(0,10):
        #     t=[]
        #     for j in range(0,10):
        #         if i==j:
        #             t.append(1)
        #         else:
        #             t.append(0)
        #         pass
        #     self.soft_10.append(t)
        #     pass

        # print("[]")
        # print(self.score_10)
        # print("[]")
        # print(self.soft_10)

        # print("-------------------------------------------------------------------------------------------------------")

        for i in range(0,100):
            id=i+1
            t=[]
            for j in range(0,100):
                it=j+1
                if it==id:
                    t.append(1.0)
                elif abs(it-id)<=5:
                    t.append(0.7)
                elif abs(it-id)<=10:
                    t.append(0.3)
                else:
                    t.append(0.0)
                pass
            self.score_100.append(t)
            pass

            self.soft_100=c(self.score_100)


        # for i in range(0,100):
        #     t=[]
        #     sum=0.0
        #     for j in range(0,100):
        #         sum+=self.score_100[i][j]
        #         pass
        #     for j in range(0,100):
        #         t.append(self.score_100[i][j]/sum)
        #         pass
        #     self.soft_100.append(t)
        #     pass

        # for i in range(0,100):
        #     t=[]
        #     for j in range(0,100):
        #         if i==j:
        #             t.append(1)
        #         else:
        #             t.append(0)
        #         pass
        #     self.soft_100.append(t)
        #     pass

        # print("[]")
        # print(self.score_100)
        # print("[]")
        # print(self.soft_100)

        # self.score_10=torch.tensor(self.score_10)
        # self.soft_10=torch.tensor(self.soft_10)
        # self.score_100=torch.tensor(self.score_100)
        # self.soft_100=torch.tensor(self.soft_100)
        pass

    pass

print("Config")

# c=Configs()
#
# print(c.soft_10[0])

