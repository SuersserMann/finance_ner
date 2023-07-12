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
        self.cc = nn.Linear(768, 614)
        # self.dropout = nn.Dropout(0.5)  # 添加dropout层
        self.crf = CRF(614)  # 添加CRF层

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
        events = item['events']

        # 过滤掉events为空的数据
        if events == []:
            return None

        return text_id, text, events


class Dataset1(data.Dataset):
    def __init__(self, filename):
        with jsonlines.open(filename, 'r') as f:
            self.data = list(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        text_id = item['text_id']
        text = item['text']

        return text_id, text


val_dataset = Dataset('dev.jsonl')
test_dataset = Dataset1('testA.jsonl')

val_dataset = [item for item in val_dataset if item is not None]

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


def get_type(index):
    if index < len(type_list):
        if index < 103:
            return type_list[index]


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


def get_labels_by_text_id(text_id, result1_data):
    for data in result1_data:
        if data["text_id"] == text_id:
            return data["labels"]
    return None


def change_type(text_id, lst):
    result1_list = get_labels_by_text_id(text_id, result1_data)
    result1_list = [num - 103 if 102 < num < 206 else num for num in result1_list]
    # for xx, item in enumerate(lst):
    #     if result1_list[xx] == 206:
    #         lst[xx] = 612
    #     elif lst[xx] - result1_list[xx] - 205 == result1_list[xx]:
    #         lst[xx] = result1_list[xx]
    # for xx, item in enumerate(lst):
    #     if item <= 205:
    #         lst[xx] = result1_list[xx]
    #     elif 205 < item < 409 and lst[xx] - result1_list[xx] - 205 >= 0:
    #         lst[xx] = [result1_list[xx], lst[xx] - result1_list[xx] - 205]
    #     elif 205 < item < 409 and lst[xx] - result1_list[xx] - 205 < 0:
    #         lst[xx] = 612
    # return lst
    for xx, item in enumerate(lst):
        if 205 < lst[xx] < 409 and result1_list[xx] != 206 and lst[xx] - result1_list[xx] - 205 >= 0:
            lst[xx] = [result1_list[xx], lst[xx] - result1_list[xx] - 205]
        else:
            lst[xx] = result1_list[xx]
    return lst


def generate_lists(text_id, lst):
    lst = change_type(text_id, lst)
    list1 = []
    list2 = []

    current_num = lst[0]
    start_index = 0

    for i in range(1, len(lst)):
        if lst[i] != current_num:
            list1.append(current_num)
            list2.append([start_index - 1, i - 2])
            current_num = lst[i]
            start_index = i

    # 处理最后一个数字
    list1.append(current_num)
    list2.append([start_index - 1, len(lst) - 2])

    # 处理嵌套的列表
    list1, list2 = flatten_nested_list(list1, list2)
    list1, list2 = remove_greater_than_102(list1, list2)
    return list1, list2


def flatten_nested_list(list1, list2):
    flattened_list1 = []
    flattened_list2 = []
    for i, item in enumerate(list1):
        if isinstance(item, list):
            flattened_list1.extend(item)
            flattened_list2.extend([list2[i]] * len(item))
        else:
            flattened_list1.append(item)
            flattened_list2.append(list2[i])
    return flattened_list1, flattened_list2


def remove_greater_than_102(list1, list2):
    new_list1 = []
    new_list2 = []

    for i in range(len(list1)):
        if list1[i] <= 102:
            new_list1.append(list1[i])
            new_list2.append(list2[i])

    return new_list1, new_list2


def get_substring(text, indices):
    start_index, end_index = indices
    substring = text[start_index:end_index + 1]
    return substring


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

        text_one = re.sub(r'[ \u3000\xa0\u2002\u2003�]+', '', text_one)
        text_one = text_one.replace('️', '')
        text_one = remove_unrecognized_unicode(text_one)

        if len(text_one) > 510:
            text_one = text_one[:510]
        characters = [char for char in text_one]

        characters_list.append(characters)
        text.append(text_one)
    data = token.batch_encode_plus(

        characters_list,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
        is_split_into_words=True,
        return_length=True)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    return input_ids, attention_mask, text_id, text


val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=16,
                                         collate_fn=collate_fn,
                                         shuffle=False,
                                         drop_last=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          collate_fn=collate_fn,
                                          shuffle=False,
                                          drop_last=False)

os.makedirs("多标签", exist_ok=True)
os.path.join("多标签")
seed_list = [4000, 50, 90, 123, 1100, 1600, 2023]
for seed in seed_list:
    model = Model(seed)
    model.to(device)
    model.load_state_dict(torch.load(f'model/model_seed_duo{seed}.pth'))
    model.eval()
    p_list = []
    t_list = []
    c_list = []
    with open(f"单标签/dan_seed_{seed}.txt", "r", encoding='utf-8') as file:
        result1_data = [json.loads(line) for line in file]
    with torch.no_grad():
        for i, (input_ids, attention_mask, text_ids, texts) in enumerate(test_loader):

            out = model(input_ids=input_ids, attention_mask=attention_mask)

            out = model.crf.viterbi_decode(out, attention_mask.to(torch.bool))
            # out = torch.where(out > threshold, torch.ones_like(out), torch.zeros_like(out))
            predicted_labels = out
            for j, predicted_label in enumerate(out):
                text_id_one = text_ids[j]
                text_events = texts[j]
                # predicted_labels.append(predicted_label)
                t_modified_label = predicted_label[0:len(text_events) + 2]
                new_p = [num - 103 if 102 < num < 206 else (num - 203 if 408 < num < 612 else num) for num in
                         t_modified_label]
                # {"type":"被列为失信被执行人" ,"entity":"播州城投"}
                list3, list4 = generate_lists(text_id_one, new_p)
                b_list = []
                for t in range(len(list3)):
                    a_list = {"type": get_type(list3[t]), "entity": get_substring(text_events, list4[t])}
                    if a_list not in b_list:
                        b_list.append(a_list)
                # {"text_id": "123456", "events":[]}
                c_list.append({"text_id": text_id_one, "events": b_list})
            print("总数据", i)
    filename = f'多标签/多_seed_{seed}.txt'
    print(f"save多标签/多_seed_{seed}.txt", i)
    save_list_to_txt(c_list)
