import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from models import BERT, BI_GRU, TEXT_CNNS

pathc="./bert_model"

#path="D:\\AI\\py_package\\toutiao_cat_data\\toutiao_cat_data.txt"

#model_dir="D:\\AI\\py_package\\bert-base-chinese"
# config = BertConfig.from_pretrained(model_dir)
# tokenizer = BertTokenizer.from_pretrained(model_dir)
# BERT_model = BertModel.from_pretrained(model_dir)

tokenizer = BertTokenizer.from_pretrained(pathc)
#BERT_model = BertModel.from_pretrained(pathc)
print("+----------------------------------------------------------------------------------------------------------------------------------------------------------------+")

# path1="D:\\AI\py_package\\toutiao_cat_data\\tou.csv"
#
# model_dir="D:\\AI\\py_package\\bert-base-chinese"
# config = BertConfig.from_pretrained(model_dir)
# tokenizer = BertTokenizer.from_pretrained(model_dir)
# model = BertModel.from_pretrained(model_dir)
#
# c="秋阴不散霜飞晚，留得残荷听雨声。真是good。"
#
# print(tokenizer.tokenize(c))
# print(tokenizer.encode_plus(c)['input_ids'])
# print(tokenizer.encode_plus(c)['token_type_ids'])
# print(tokenizer.encode_plus(c)['attention_mask'])
#
# print(tokenizer.convert_ids_to_tokens(tokenizer.encode_plus(c)['input_ids']))

#print(tokenizer.encode(c))
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate=0.00001
epoch=3
batch_size=32
max_len=64
word_size=21128
hn=768
num_class=10


path_text='./datas/text.txt'
print("+----------------------------------------------------------------------------------------------------------------------------------------------------------------+")

def build_dataset_with_txt(path,max_len):
    ltmp=[]
    bad=0
    with open(path, 'r',encoding='utf-8-sig') as fp :
        ll = fp.readlines()
        tot = len(ll)
        print("tot = ",tot)
        u=0
        for l in ll:
            u+=1
            #print(u)
            if u%2000==0:
                print(" u = ",u)
                pass
            ww = l.split(',')
            if len(ww)!=5:
                bad+=1
                continue
                pass
            data= ww[0]
            p1= int(ww[1])
            p2= int(ww[2])
            p3= int(ww[3])
            p4= int(ww[4])
            ltmp.append((data,(p1,p2,p3,p4)))
            pass
        pass
    print('size = ',len(ltmp))

    print('[---------------------------------------------------------------------------------------------------------]')
    print("bad = ",bad)
    # path = "./datas/train_data.xlsx"
    # wbb = openpyxl.load_workbook(path)
    # sheet_lis1 = wbb.get_sheet_names()
    # wb = wbb[sheet_lis1[0]]
    #
    # print("sheet title:", wb.title)
    # print("sheet rows:", wb.max_row)
    # print("sheet column:", wb.max_column)
    #
    # for i in range(2,wb.max_row+1):
    #     data= wb[i][8].value
    #     p1= float(wb[i][9].value)
    #     p2= int(wb[i][10].value)
    #     p3= int(wb[i][11].value)
    #     p4= int(wb[i][12].value)
    #
    #     ltmp.append((data,(p1,p2,p3,p4)))
    #     pass

    print('-----------------------datas loading have finished-----------------------')

    contents_train=[]
    contents_test=[]
    for i in range(0,20000):
        x,y=ltmp[i]
        token=tokenizer.convert_ids_to_tokens(tokenizer.encode_plus(x)['input_ids'])
        seq_len=len(token)
        mask=[]
        token_ids=tokenizer.convert_tokens_to_ids(token)
        if len(token_ids)<max_len:
            mask=[1]*seq_len+[0]*(max_len-seq_len)
            token_ids+=([0]*(max_len-seq_len))
            pass
        elif len(tokenizer)>max_len:
            mask=[1]*max_len
            token_ids=token_ids[:max_len]
            seq_len=max_len
            pass
        else:
            mask=[1]*max_len
            token_ids=token_ids
            seq_len=max_len
            pass
        contents_train.append((token_ids,y,seq_len,mask))
        pass
    for i in range(20000,20000+2000):
        x, y = ltmp[i]
        token=tokenizer.convert_ids_to_tokens(tokenizer.encode_plus(x)['input_ids'])
        seq_len=len(token)
        mask=[]
        token_ids=tokenizer.convert_tokens_to_ids(token)
        if len(token_ids)<max_len:
            mask=[1]*seq_len+[0]*(max_len-seq_len)
            token_ids+=([0]*(max_len-seq_len))
            pass
        elif len(tokenizer)>max_len:
            mask=[1]*max_len
            token_ids=token_ids[:max_len]
            seq_len=max_len
            pass
        else:
            mask=[1]*max_len
            token_ids=token_ids
            seq_len=max_len
            pass
        contents_test.append((token_ids,y,seq_len,mask))
        pass

    return contents_train,contents_test
    pass

print('+-------------------------------------------------------------------------------------------------------------+')
class Dataset_iterater():
    def __init__(self,batchs,batch_size,device):
        self.batchs=batchs
        self.batch_size=batch_size
        self.num_b=len(batchs)//batch_size
        self.res=False
        if len(batchs)%self.batch_size!=0:
            self.res=True
            pass
        self.inde=0
        self.device=device
        pass
    def _to_tensor(self,datas):
        x=torch.LongTensor([i[0] for i in datas]).to(self.device)
        y=torch.LongTensor([i[1] for i in datas]).to(self.device)
        seq_len=torch.LongTensor([i[2] for i in datas]).to(self.device)
        mask=torch.LongTensor([i[3] for i in datas]).to(self.device)
        return (x,seq_len,mask),y
        pass

    def __next__(self):
        if self.res==True and self.inde==self.num_b:
            batchs=self.batchs[self.inde*self.batch_size:len(self.batchs)]
            self.inde+=1
            batchs=self._to_tensor(batchs)
            return batchs
            pass
        elif self.inde>=self.num_b:
            self.inde=0
            raise StopIteration
            pass
        else:
            batchs=self.batchs[self.inde*self.batch_size:(self.inde+1)*self.batch_size]
            self.inde+=1
            batchs=self._to_tensor(batchs)
            return batchs
            pass
        pass
    def __iter__(self):
        return self
    def __len__(self):
        if self.res==True:
            return self.num_b+1
            pass
        else:
            return self.num_b
            pass
        pass
    pass

print('+[net]-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+')

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.my_bert= BERT.BERT_Net()
        self.my_text_cnn= TEXT_CNN.TEXTCNN_Net()
        self.my_bi_gru= BI_GRU.BI_GRU_Net()

        self.L_possibility=torch.nn.Linear(hn*3,100)
        self.L_seriousness=torch.nn.Linear(hn*3,num_class)
        self.L_Risk_impact_factors=torch.nn.Linear(hn*3,num_class)
        self.L_Risk_diffusion_degree=torch.nn.Linear(hn*3,num_class)

        pass
    def forward(self,x,hidden,num_b):
        out1=self.my_bert(x)
        out2=self.my_text_cnn(x[0])
        out3=self.my_bi_gru(x[0],hidden,num_b)
        out=[out1,out2,out3]
        out=torch.cat(out,dim=1)

        possibility=self.L_possibility(out)
        seriousness=self.L_seriousness(out)
        Risk_impact_factors=self.L_Risk_impact_factors(out)
        L_Risk_diffusion_degree=self.L_Risk_diffusion_degree(out)

        return (possibility,seriousness,Risk_impact_factors,L_Risk_diffusion_degree)
        pass
    pass





net=Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

contents_train,contents_test=build_dataset_with_txt(path_text,max_len)
train_iter=Dataset_iterater(contents_train,batch_size,device)
test_iter=Dataset_iterater(contents_test,batch_size,device)

print("----------------------training start----------------------")
def f_train():
    net.train()
    for i,(datas,labels) in enumerate(train_iter):

        net.zero_grad()
        num_b = datas[0].size(0)
        h0=net.my_bi_gru.inithidden(num_b)
        outputs=net(datas,h0,num_b)
        p1,p2,p3,p4=outputs

        y1=torch.LongTensor([i[0] for i in labels]).to(device)
        y2=torch.LongTensor([i[1] for i in labels]).to(device)
        y3=torch.LongTensor([i[2] for i in labels]).to(device)
        y4=torch.LongTensor([i[3] for i in labels]).to(device)

        # y1 =[label[0] for label in labels]
        # y2 =[label[1] for label in labels]
        # y3 =[label[2] for label in labels]
        # y4 =[label[3] for label in labels]

        #y1=torch.tensor(y1).to(device)
        loss1=F.cross_entropy(p1,y1)
        loss2=F.cross_entropy(p2,y2)
        loss3=F.cross_entropy(p3,y3)
        loss4=F.cross_entropy(p4,y4)

        loss=loss1+loss2+loss3+loss4
        loss.backward()
        optimizer.step()
        if i%100==0:
            print("i = ",i," [] loss = ",loss)
            print("loss : ",(loss1).item()," ",loss2.item()," ",loss3.item()," ",loss4.item())
            pass
        pass
    pass

def f_test():
    net.eval()
    # correct=torch.Tensor([0]).to(device)
    # total=torch.Tensor([0]).to(device)
    with torch.no_grad():
        for i,(datas,labels) in enumerate(test_iter):
            net.zero_grad()
            num_b = datas[0].size(0)
            h0 = net.my_bi_gru.inithidden(num_b)
            outputs = net(datas, h0, num_b)

            y1 = torch.LongTensor([i[0] for i in labels]).to(device)
            y2 = torch.LongTensor([i[1] for i in labels]).to(device)
            y3 = torch.LongTensor([i[2] for i in labels]).to(device)
            y4 = torch.LongTensor([i[3] for i in labels]).to(device)
            p1, p2, p3, p4 = outputs


            # print("p1 ",p1.shape)
            # print("p2 ",p2.shape)
            p1 = torch.max(p1, 1)[1]
            p2 = torch.max(p2, 1)[1]
            p3 = torch.max(p3, 1)[1]
            p4 = torch.max(p4, 1)[1]



            #only output the prediction value of first example in each batch
            print("label : ",y1[0].item()," ",y2[0].item()," ",y3[0].item()," ",y4[0].item())
            print("predict : ",(p1[0]).item()," ",p2[0].item()," ",p3[0].item()," ",p4[0].item())


            pass
        pass
    # print(correct)
    # print(total)
    #print(' accuracy  :{} '.format(100*correct[0]/total[0]))
    pass

print('+-------------------------------------------------------------------------------------------------------------+[start]')

for i in range(epoch):
    print("epoch  : ",i)
    f_train()
    f_test()
    print("--------------------------------------------------")
    pass
















print("+-------------------------------------------------------------------------------------------------------------------------------------+[end]")




