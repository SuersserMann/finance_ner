import os
import random
import re
import unicodedata
import numpy as np
import torch
from TorchCRF import CRF
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.utils.data as Data
import json
import torch.utils.data as data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import torch.nn.functional as F
import jsonlines
import regex

warnings.filterwarnings("ignore")

print(torch.__version__)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)


class Model(torch.nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.pretrained = AutoModel.from_pretrained("bert-base-chinese/")
        self.pretrained.to(device)
        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        self.lstm = nn.LSTM(768, 384, num_layers=2, batch_first=True, bidirectional=True)
        self.cc = nn.Linear(768, 208)
        # self.dropout = nn.Dropout(0.5)  # 添加dropout层
        self.crf = CRF(208)  # 添加CRF层

    def forward(self, input_ids, attention_mask):
        out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        out = out.last_hidden_state
        out, _ = self.lstm(out)
        # out = self.dropout(out)  # 应用dropout层
        out = self.cc(out)
        return out


model = Model()
model.to(device)


class Dataset(data.Dataset):
    def __init__(self, filename):
        with jsonlines.open(filename, 'r') as f:
            self.data = list(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        text_id = item['text_id']
        text = item['text']
        # events = item['events']

        # 过滤掉events为空的数据
        # if events == []:
        #     return None

        return text_id, text


test_dataset = Dataset('testA.jsonl')

token = AutoTokenizer.from_pretrained("bert-base-chinese/")

type_list = ['公司注销', '高层涉嫌违法', '高层变更', '关闭分支机构', '吊销资质牌照', '经营期限到期', '造假欺诈',
             '偷税漏税', '信息泄露', '重大赔付', '窃取别人商业机密', '违规催收', '网站安全漏洞', '实际控制人涉诉仲裁',
             '实际控制人违规', '实际控制人变更', '盗取隐私信息', '破产清算', '警告', '重大资产损失', '财务信息造假',
             '实际控制人涉嫌违法', '债务违约', '欠薪', '外部信用评级下调', '内幕交易', '偿付能力不足',
             '评级机构中止评级', '公司停牌', '债务融资失败', '资金紧张', '债务展期', '债务重组', '资产质量下降',
             '资本充足不足', '重大债务到期', '被银行停贷', '停产停业', '盈利能力下降', '高层失联/死亡', '延期信息披露',
             '资产冻结', '经营亏损', '投资亏损', '裁员', '股权冻结/强制转让', '澄清辟谣', '退出市场', '公司退市',
             '吊销业务许可或执照', '业务/资产重组', '连续下跌', '实际控制人失联/死亡', '企业被问询约谈审查', '挤兑',
             '员工罢工示威', '发放贷款出现坏账', '被接管', '股东利益斗争', '监管入驻', '产品违约/不足额兑付',
             '基层员工流失', '更换基金经理', '第一大股东变化', '履行连带担保责任', '重大安全事故', '经营激进',
             '无法表示意见', '股票发行失败', '保留意见', '出具虚假证明', '暂停上市', '责令改正', '禁入行业',
             '限制业务范围', '停止接受新业务', '产品虚假宣传', '公司违规关联交易', '非法集资', '扰乱市场秩序',
             '终身禁入行业', '撤销任职资格', '税务非正常户', '大量投诉', '总部被警方调查', '被列为失信被执行人',
             '分支机构被警方调查', '骗保', '监管评级下调', '财务报表更正', '否定意见', '自然灾害', '行业排名下降',
             '限制股东权利', '股权查封', '签订对赌协议', '审计师辞任', '股权融资失败', '停止批准增设分支机构',
             '薪酬福利下降', '误操作', '授信额度减少', '经营资质瑕疵']


def remove_unrecognized_unicode(text):
    cleaned_text = regex.sub(r'\p{C}', '', text)
    return cleaned_text


def find_indices(cfn_spans_start, word_start):
    matches = []
    for i, num in enumerate(word_start):
        if num in cfn_spans_start:
            matches.append(i)
    return matches


def reshape_and_remove_pad(outs, labels, attention_mask):
    outs = outs[attention_mask == 1]

    # Reshape 'labels' tensor based on attention_mask
    labels = labels[attention_mask == 1]

    return outs, labels


def get_index(target_type):
    return type_list.index(target_type)


def find_all(text, sub):
    start = 0
    while start < len(text):
        start = text.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def custom_encoder(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def save_list_to_txt(data_list):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data_list:
            json_data = json.dumps(item, ensure_ascii=False, default=custom_encoder)
            file.write(json_data + '\n')


def collate_fn(data):
    text_id = []
    text = []
    events = []
    labels = []
    characters_list = []
    for i in data:
        text_id.append(i[0])
        text_one = i[1]
        text.append(i[1])
        text_one = re.sub(r'[ \u3000\xa0\u2002\u2003�]+', '', text_one)
        text_one = text_one.replace('️', '')
        text_one = remove_unrecognized_unicode(text_one)

        if len(text_one) > 510:
            text_one = text_one[:510]
        characters = [char for char in text_one]

        characters_list.append(characters)


    data = token.batch_encode_plus(

        characters_list,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
        is_split_into_words=True,
        return_length=True)

    lens = data['input_ids'].shape[1]

    for i in range(len(labels)):

        labels[i] = [207] + labels[i]
        labels[i] += [207] * lens
        labels[i] = labels[i][:lens]

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)


    return input_ids, attention_mask, text_id


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          collate_fn=collate_fn,
                                          shuffle=False,
                                          drop_last=False)
os.makedirs("单标签", exist_ok=True)
os.path.join("单标签")
seed_list = [4000, 50, 90, 123, 1100, 1600, 2023]

for seed in seed_list:
    model = Model(seed)
    model.to(device)
    model.load_state_dict(torch.load(f'model/model_seed_{seed}.pth'))
    model.eval()

    p_list = []
    t_list = []
    c_list = []
    a = 0
    with torch.no_grad():
        for i, (input_ids, attention_mask, text_id) in enumerate(test_loader):

            out = model(input_ids=input_ids, attention_mask=attention_mask)

            out = model.crf.viterbi_decode(out, attention_mask.to(torch.bool))
            # out = torch.argmax(out, dim=2)
            # out = torch.where(out > threshold, torch.ones_like(out), torch.zeros_like(out))

            predicted_labels = out

            for gg in range(len(out)):
                yy_list = {"text_id": text_id[gg], "labels": predicted_labels[gg]}
                c_list.append(yy_list)
                a += 1
            print(a)
    filename = f'单标签/dan_seed_{seed}.txt'
    save_list_to_txt(c_list)
