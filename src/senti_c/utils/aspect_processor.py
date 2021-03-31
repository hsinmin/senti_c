# coding=utf-8
# Some of structures or codes in this script are referenced from HuggingFace Inc. team. 

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''屬性情感分類的資料讀取與處理'''

import os
import copy
import csv
import json
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

ASPECT_LABEL_MAP = {0:"B", 1:"I", 2:"O"}
SENTIMENT_LABEL_MAP = {0:"NEG",1:"NEU",2:"POS",3:"O"}


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label=None,evaluate_label_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        
        self.evaluate_label_ids = evaluate_label_ids 


class AspectProcessor:
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
        
    def get_train_dev_examples(self, data_dir, mode):
        file_path = os.path.join(data_dir, mode+".txt")
        return self._create_examples(file_path, mode)

    def get_test_examples_from_list(self, lines,finish_word_seg):
        '''data為list格式'''
        return self._create_examples_from_test(lines,finish_word_seg)
    
    def get_test_examples_from_tsv(self, data_dir):
        '''這裡的 data_dir 為使用者自己指定的檔名(路徑)'''
        
        ## 讀取tsv檔案
        data = self._read_tsv(data_dir) #得到一個大list，裡面每個list中包含tsv檔案中的一列資料
        
        ## 將資料整理成一個list,裡面每個元素都是一個String
        new_data = [i[0] for i in data] #因為只有文本所以取[0] 
        
        return new_data

    def get_labels(self):
        return ['B-NEG', 'B-NEU', 'B-POS', 'I-NEG', 'I-NEU', 'I-POS', 'O']   
    
    def _create_examples_from_test(self, lines, finish_word_seg):
        '''這是給沒有label的test資料所使用；lines型別為list'''
        
        examples = []
        all_data_words = [] #保存每句話對應的要預測的字
        for (i, line) in enumerate(lines):
            guid = "test-%s" % (i)
            
            if finish_word_seg: #使用者已用空白隔開各字，每個字就是要預測的內容
                text_a = line.split(" ")     
            else:
                text_a = list(line)
                
            text_a = [tmp for tmp in text_a if (len(tmp) > 0 and not tmp.isspace())] #過濾空白的字
            labels = [None] * len(text_a)
            
            all_data_words.append(text_a)
     
            examples.append(InputExample(guid=guid, words=text_a, labels=labels))
        return examples,all_data_words
    
    def _create_examples(self,file_path, mode):
        '''創建訓練和驗證時的資料examples'''
        
        guid_index = 0
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))

            if words:
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
        return examples

    
    
def aspect_convert_examples_to_features(
    examples,
    tokenizer,
    label_list=None,
    max_length=128,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    mask_padding_with_zero=True,
    logging_display=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Returns:
        If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    
    label_map = {label: i for i, label in enumerate(label_list)}
# label map:  {'B-NEG': 0, 'B-NEU': 1, 'B-POS': 2, 'I-NEG': 3, 'I-NEU': 4, 'I-POS': 5, 'O': 6}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and logging_display:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        evaluate_label_ids = []  #該句子中實際需要去預測/計算的位置 (等於非特殊符號或subword的位置)
        valid_idx = 0 
        
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if label is not None:
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            else:
                label_ids.extend([None] + [pad_token_label_id] * (len(word_tokens) - 1))
                
            evaluate_label_ids.append(valid_idx)
            valid_idx += len(word_tokens)    
        
        assert valid_idx == len(tokens)
        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        
        special_tokens_count = 2
        if len(tokens) > max_length - special_tokens_count:
            tokens = tokens[: (max_length - special_tokens_count)]
            label_ids = label_ids[: (max_length - special_tokens_count)]

            
        tokens += ["[SEP]"]
        label_ids += [pad_token_label_id]
        segment_ids = [0] * len(tokens)

        tokens = ["[CLS]"] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [0] + segment_ids
        evaluate_label_ids += 1  #因為最前面加了"CLS"，所以索引都加1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)
        
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length
        assert len(label_ids) == max_length

        if ex_index < 3 and logging_display:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            if label_ids[1] is not None:
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label=label_ids,evaluate_label_ids=evaluate_label_ids)
        )
    

    return features


def match_ot(gold_ote_sequence, pred_ote_sequence):
    
    n_hit = 0
    for t in pred_ote_sequence:
        if t in gold_ote_sequence:
            n_hit += 1
    return n_hit


def match_ts(gold_ts_sequence, pred_ts_sequence):

    tag2tagid = {'NEG': 0, 'NEU': 1, 'POS': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    
    for t in gold_ts_sequence:
        ts_tag = t[2]  #拿到該pair的情感
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
        
    return hit_count, gold_count, pred_count


def tag2ot(ote_tag_sequence):
    '''從輸入的屬性 BIO tag 中取出每個屬性對應的開始與結束位置索引'''
    
    n_tags = len(ote_tag_sequence)
    ot_sequence = []
    beg, end = -1, -1
    checkafters = []  #每個位置會放該位置後面有幾個
    for i in range(n_tags):
        tag = ote_tag_sequence[i]
        
        if tag == 'B':  #只抓取從'B'開始的才納為屬性
            beg = i
            end = i  #預設為i，避免這個位置是整句話中最後一個位置
            
            for j in range(i+1,n_tags):   #從該indexs後面開始，如果i+1 = n_tags , 則不會進入此區間 (表示i為這句話中的最後一個位置)
                if ote_tag_sequence[j] == 'O' or ote_tag_sequence[j] == 'B':  #表示後面接O或B
                    end = j-1
                    break
                elif ote_tag_sequence[j] == 'I':
                    end = j  #先紀錄，避免這個位置(j)為整句話中最後一個字
            
            if end >= beg > -1:
                ot_sequence.append((beg, end))
                beg, end = -1, -1
            
    return ot_sequence


def tag2ts(ts_tag_sequence):
    '''從輸入的屬性-情感 tag 中取出每個屬性對應的開始與結束位置索引與情感，注意只有同一屬性中每個字情感皆一致才提取'''
    
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        
        # current position and sentiment
        eles = ts_tag.split('-')

        pos, sentiment = eles
        
        if pos == 'B':  #抓取從'B'開始的
            beg = i
            end = i  #預設為i，避免這個位置是整句話中最後一個位置
            sentiments.append(sentiment)  #將其對應情感加入
            
            for j in range(i+1,n_tags):   #從該indexs後面開始，如果i+1 = n_tags , 則不會進入此區間 (表示i為這句話中的最後一個位置)
                tmp = ts_tag_sequence[j].split('-')
                
                if tmp[0] == 'B' or tmp[0] == 'O':  #表示後面接O或B
                    end = j-1
                    break
                elif tmp[0] == 'I': 
                    sentiments.append(tmp[1])
                    end = j  #先紀錄，避免這個位置(j)為整句話中最後一個字
    
            # 該屬性中每個字的情感都一致時才會納入
            # 並避免遇到屬性有預測(有找到(start,end))、但情感部分預測為O (有此狀況則不納入)
            if end >= beg > -1 and len(set(sentiments)) == 1 and sentiments[0] != 'O':  
                ts_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1
    
    return ts_sequence

def evaluate_ote(gold_ot, pred_ot):
    """
    evaluate the model performance for the ote task
    :param gold_ot: gold standard ote tags 
    :param pred_ot: predicted ote tags
    """
    
    assert len(gold_ot) == len(pred_ot)
    
    n_samples = len(gold_ot)
    
    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    
    for i in range(n_samples):  
        g_ot = gold_ot[i]  #對每筆資料來做
        p_ot = pred_ot[i]
      
        g_ot_sequence, p_ot_sequence = tag2ot(ote_tag_sequence=g_ot), tag2ot(ote_tag_sequence=p_ot)

        
        # 計算 hit number
        n_hit_ot = match_ot(gold_ote_sequence=g_ot_sequence, pred_ote_sequence=p_ot_sequence)
        n_tp_ot += n_hit_ot
        n_gold_ot += len(g_ot_sequence)
        n_pred_ot += len(p_ot_sequence)
    
    ot_precision = float(n_tp_ot) / float(n_pred_ot + 0.001)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + 0.001)
    ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + 0.001)
    
    ote_scores = {"aspect_precision": ot_precision, 
                  "aspect_recall": ot_recall, 
                  "aspect_f1": ot_f1}
    
    return ote_scores


def evaluate_ts(gold_ts, pred_ts):
    """
    evaluate the model performance for the ts task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    """
    
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)

    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)

    for i in range(n_samples):
        g_ts = gold_ts[i]
        p_ts = pred_ts[i]
     
        g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=g_ts), tag2ts(ts_tag_sequence=p_ts)
        
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)
        
        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        
    for i in range(3):  #有三類別
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        
        ts_precision[i] = float(n_ts) / float(n_p_ts + 0.001)
        ts_recall[i] = float(n_ts) / float(n_g_ts + 0.001)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + 0.001)

    ts_macro_f1 = ts_f1.mean()  #算macro f1

    n_tp_total = sum(n_tp_ts)
    n_g_total = sum(n_gold_ts)
    n_p_total = sum(n_pred_ts)

    ts_micro_p = float(n_tp_total) / (n_p_total + 0.001)
    ts_micro_r = float(n_tp_total) / (n_g_total + 0.001)
    ts_micro_f1 = 2 * ts_micro_p * ts_micro_r / (ts_micro_p + ts_micro_r + 0.001)
    
    ts_scores = {"absa_macro_neg": ts_f1[0],
                 "absa_macro_neu": ts_f1[1],
                 "absa_macro_pos": ts_f1[2], 
                 "absa_macro_f1": ts_macro_f1,
                 "absa_micro_precision": ts_micro_p, 
                 "absa_micro_recall": ts_micro_r, 
                 "absa_micro_f1": ts_micro_f1}
    
    return ts_scores
                                    


def chg_labels_to_aspect_and_sentiment(input_labels):
    '''將原本合併的屬性-情感標記轉成分開的形式'''
    
    aspect_labels =  input_labels.clone().detach()  
    sentiment_labels =  input_labels.clone().detach()
    
    for i,j in enumerate(input_labels):  #每個j都是該batch中的一筆資料
        for m,n in enumerate(j):  #每個n都是一筆資料中的一個token
            if n == 0:  # 'B-NEG'
                aspect_labels[i][m] = 0
                sentiment_labels[i][m] = 0
            if n == 1:  # 'B-NEU'
                aspect_labels[i][m] = 0
                sentiment_labels[i][m] = 1    
            if n == 2:  # 'B-POS'
                aspect_labels[i][m] = 0
                sentiment_labels[i][m] = 2
            if n == 3:  # 'I-NEG'
                aspect_labels[i][m] = 1
                sentiment_labels[i][m] = 0   
            if n == 4:  # 'I-NEU'
                aspect_labels[i][m] = 1
                sentiment_labels[i][m] = 1
            if n == 5:  # 'I-POS'
                aspect_labels[i][m] = 1
                sentiment_labels[i][m] = 2
            if n == 6:  # 'O'
                aspect_labels[i][m] = 2
                sentiment_labels[i][m] = 3
            if n == -100 :  
                aspect_labels[i][m] = -100
                sentiment_labels[i][m] = -100        
    
    return aspect_labels,sentiment_labels