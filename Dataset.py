import torch
from transformers import BertConfig, BertTokenizer, BertModel


class Dataset_iterater():
    def __init__(self,batchs,configs):
        self.batchs=batchs
        self.batch_size=configs.batch_size
        self.num_b=len(batchs)//(configs.batch_size)
        self.res=False
        if len(batchs)%self.batch_size!=0:
            self.res=True
            pass
        self.inde=0
        self.device=configs.device
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






class Dataset_build:
    def __init__(self,configs):
        self.path_text=configs.path_text
        self.path_bert=configs.path_bert
        self.max_len=configs.max_len
        self.tokenizer = configs.tokenizer
        pass

    def build_dataset_with_txt(self):
        ltmp = []
        with open(self.path_text, 'r', encoding='utf-8-sig') as fp:
            ll = fp.readlines()
            tot = len(ll)
            print("tot_ltmp = ", tot)
            u = 0
            for l in ll:
                u += 1
                # print(u)
                if u % 2000 == 0:
                    print(" u = ", u)
                    pass
                ww = l.split(',')
                if len(ww) != 5:
                    continue
                    pass
                data = ww[0]
                p1 = int(ww[1])
                p2 = int(ww[2])
                p3 = int(ww[3])
                p4 = int(ww[4])
                ltmp.append((data, (p1, p2, p3, p4)))
                pass
            pass
        print('size = ', len(ltmp))

        print('-----------------------datas loading have finished-----------------------')

        contents_train = []
        contents_test = []
        for i in range(0, 20000):##############
            x, y = ltmp[i]
            token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode_plus(x)['input_ids'])
            seq_len = len(token)
            mask = []
            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            if len(token_ids) < self.max_len:
                mask = [1] * seq_len + [0] * (self.max_len - seq_len)
                token_ids += ([0] * (self.max_len - seq_len))
                pass
            elif len(self.tokenizer) > self.max_len:
                mask = [1] * self.max_len
                token_ids = token_ids[:self.max_len]
                seq_len = self.max_len
                pass
            else:
                mask = [1] * self.max_len
                token_ids = token_ids
                seq_len = self.max_len
                pass
            contents_train.append((token_ids, y, seq_len, mask))
            pass
        for i in range(20000, 20000 + 2100):
            x, y = ltmp[i]
            token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode_plus(x)['input_ids'])
            seq_len = len(token)
            mask = []
            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            if len(token_ids) < self.max_len:
                mask = [1] * seq_len + [0] * (self.max_len - seq_len)
                token_ids += ([0] * (self.max_len - seq_len))
                pass
            elif len(self.tokenizer) > self.max_len:
                mask = [1] * self.max_len
                token_ids = token_ids[:self.max_len]
                seq_len = self.max_len
                pass
            else:
                mask = [1] * self.max_len
                token_ids = token_ids
                seq_len = self.max_len
                pass
            contents_test.append((token_ids, y, seq_len, mask))
            pass

        return contents_train, contents_test
        pass

    pass


print("----import Dataset----")