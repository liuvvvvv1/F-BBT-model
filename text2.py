import copy

import numpy as np
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
c="秋阴不散霜飞晚，留得残荷听雨声。真是good。"
c1='liiiu 8️⃣ ➕ ☝️ ✔ ✘ ♨️ ❄️ 【 】 ‼️ 0    ①    ij'

c2='梦话1⃣️昨天下夜班睡着了'
c3='md被电热毯烫醒'
def g(c):
    print("1 : ",tokenizer.tokenize(c))
    print("2 : ",tokenizer.encode_plus(c)['input_ids'])
    print("3 : ",tokenizer.encode_plus(c)['token_type_ids'])
    print("4 : ",tokenizer.encode_plus(c)['attention_mask'])
    print("5 : ",tokenizer.convert_ids_to_tokens(tokenizer.encode_plus(c)['input_ids']))
    print("6 : ",tokenizer.encode(c))
    pass




g(c)