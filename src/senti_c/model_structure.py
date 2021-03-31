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
'''放置句子情感分類、屬性情感分析模型架構'''   

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn import NLLLoss
from transformers import (
    BertModel, 
    BertPreTrainedModel,
)    
import numpy as np
import os
import torch.nn.functional as F 

from .utils import chg_labels_to_aspect_and_sentiment,get_domain_embedding

class MyNewBertForSequenceClassification(BertPreTrainedModel):
    '''BERT輸出接上一層線性層；此外額外加入domain embedding layer，等於最後BERT輸出和domain embedding連接、饋入分類層，是句子情感分類的模型'''
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        ### 加入 domain embedding layer
        file_path = os.path.split(os.path.realpath(__file__))[0] + "/utils/new_restaurant_emb.vec.npy"
        get_domain_embedding(file_path)
        domain_emb = np.load(file_path) 
        self.domain_embedding = nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = nn.Parameter(torch.from_numpy(domain_emb), requires_grad=True)  
        
        self.first_classifier = nn.Linear(config.hidden_size+domain_emb.shape[1],config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        ### 加入 domain embedding ### 
        input_embedding = self.domain_embedding(input_ids) 
    

        ### 將每筆資料中每個字的embedding拼起來平均 ###
        sent_embedding = torch.mean(input_embedding,dim=1)  #針對seq部分做平均
        
        ### 將平均後的embedding和bert的CLS位置輸出連接 ###
        combine_emb = torch.cat((sent_embedding,pooled_output),dim=1)

        ### 經過兩層線性層、中間添加tanh活化函數 ###
        firsts = self.first_classifier(combine_emb)
        firsts = torch.tanh(firsts)
        logits = self.classifier(firsts)

        outputs = (logits,) + outputs[2:]  

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()  #改成多標籤分類(每個標籤做二元分類的感覺)
            labels = labels.float()  
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

            outputs = (loss,) + outputs

        return outputs  


# class MyNewBertForSequenceClassification(BertPreTrainedModel):
#     '''BERT輸出接上一層線性層，是句子情感分類的模型'''
    
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             loss_fct = BCEWithLogitsLoss()  #改成多標籤分類(每個標籤做二元分類的感覺)
#             labels = labels.float()  
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
#             outputs = (loss,) + outputs 

#         return outputs  


class MyNewBertForTokenClassification(BertPreTrainedModel):
    '''將BERT每個位置輸出接上線性層預測屬性，並輔以屬性轉移矩陣設置；另外將BERT輸出額外接上另一線性層預測情感，之後用情感轉移矩陣獲得最終情感預測結果，是屬性情感分析的模型'''
    
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        ### 將BERT輸出傳到這層來做屬性標記預測 ###
        self.aspect_classfier = nn.Linear(config.hidden_size,3)  #768 -> 3 (BIO) 
        
        
        ### 增加屬性轉移矩陣 ###
#         transition_path = {'B': ['B','I', 'O'],
#                                 'I': ['B', 'I', 'O'],
#                                 'O': ['B','O']}
        self.transition_scores_aspect = [[1/3,1/3,1/3],[1/3,1/3,1/3],[1/2,0,1/2]]  #3*3矩陣，為B/I/O到B/I/O的機率     
        self.transition_scores_aspect = np.array(self.transition_scores_aspect, dtype='float32').transpose()  
        
        
        ### 增加情感轉移矩陣 (FOR 屬性標籤預測情感標籤) ###
#         transition_path = {'B': ['NEG','NEU','POS'],
#                                 'I': ['NEG','NEU','POS'],
#                                 'O': ['O']}   
        self.transition_scores_sentiment = [[1/3,1/3,1/3,0],[1/3,1/3,1/3,0],[0,0,0,1]]  #3*4矩陣，為B/I/O到NEG/NEU/POS/O的機率     
        self.transition_scores_sentiment = np.array(self.transition_scores_sentiment, dtype='float32').transpose()  
          
        
        ### 將BERT輸出傳到這層來預測情感標記 ###
        self.sentiment_classifier = nn.Linear(config.hidden_size, 4) #768 -> 4 (POS/NEU/NEG/O)
        
        self.init_weights()

        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
                 
        tagger_input = outputs[0]
        tagger_input = self.dropout(tagger_input)

        #### 以下開始對屬性部分做處理 ####
        aspect_tagger_pred = self.aspect_classfier(tagger_input)  
        
        pred_aspect = F.softmax(aspect_tagger_pred,dim=2)  
        
        if torch.cuda.is_available():
            self.transitions_aspect = torch.tensor(self.transition_scores_aspect).cuda() 
        else:
            self.transitions_aspect = torch.tensor(self.transition_scores_aspect) 
            
        combined_aspect_pred = []  
        
        for bsz in range(pred_aspect.size(0)) :  
            current_sample = pred_aspect[bsz]  
            
            current_sample_transition = []  
            for t in range(current_sample.size(0)):  
                if t == 0 or t == 1:  #對 cls & 對 cls後的第一個字 => 兩者都不用做轉移矩陣運算
                    current_sample_transition.append(current_sample[t]) 

                else:
                    transition_pred = torch.mm(self.transitions_aspect,current_sample_transition[t-1].view(3,-1))  #轉移矩陣乘上前一個token已經權衡相加後的值
                   
                    
                    ### 計算 confidence value (用轉移矩陣算出的值的重要程度) ###
                    val = torch.sum(current_sample_transition[t-1].view(3,-1) * current_sample_transition[t-1].view(3,-1)) 
                    alpha = val * 0.6  
                    new_aspect_pred = alpha * transition_pred + (1-alpha) * current_sample[t].view(3,-1) 
                    current_sample_transition.append(new_aspect_pred.view(-1))  

            combined_aspect_pred.append(torch.stack(current_sample_transition, 0))   
             
        aspect_tagger = torch.stack(combined_aspect_pred, 0)  

        
        ### 以下開始對情感部分做處理 ###
        sentiment_pred = self.sentiment_classifier(tagger_input) 
        
        if torch.cuda.is_available():
            self.transitions_sentiment = torch.tensor(self.transition_scores_sentiment).cuda()
        else:
            self.transitions_sentiment = torch.tensor(self.transition_scores_sentiment)
        
        sentiment_pred = F.softmax(sentiment_pred,dim=2)  
        
        combined_sentiment_pred = []  
        
        for bsz in range(sentiment_pred.size(0)) :  
            current_sample_aspect = aspect_tagger[bsz]  
            current_sample_sentiment = sentiment_pred[bsz] 
            
            current_sample_transition = []  
            for t in range(current_sample_sentiment.size(0)): 
                if t == 0:  #對cls不用做轉移矩陣運算
                    current_sample_transition.append(current_sample_sentiment[t])  
                else:
                    transition_pred = torch.mm(self.transitions_sentiment,current_sample_aspect[t].view(3,-1))  #轉移矩陣乘上屬性預測標籤
                   
                    ### 計算 confidence value (這個用轉移矩陣算出的值的重要程度) ###  
                    val = torch.sum(current_sample_aspect[t].view(3,-1) * current_sample_aspect[t].view(3,-1)) 
                    
                    alpha = val * 0.6 
                    
                    new_sentiment_pred = alpha * transition_pred + (1-alpha) * current_sample_sentiment[t].view(4,-1) 
                    current_sample_transition.append(new_sentiment_pred.view(-1))  

            combined_sentiment_pred.append(torch.stack(current_sample_transition, 0))    
             
        logits = torch.stack(combined_sentiment_pred, 0)  
        
        
        outputs = (aspect_tagger,) + (logits,) + outputs[2:]   
        
    
        ### 輸入的label map:  {'B-NEG': 0, 'B-NEU': 1, 'B-POS': 2, 'I-NEG': 3, 'I-NEU': 4, 'I-POS': 5, 'O': 6}
        ## 自定義的新label map : 
            # aspect_labels : {"B" : 0, "I":1, "O": 2}
            # sentiment_labels {"NEG": 0 ,"NEU":1, "POS":2, "O":3}
        
        if labels is not None:
            #處理 labels、讓它符合joint屬性模型格式 
            aspect_labels, sentiment_labels = chg_labels_to_aspect_and_sentiment(labels)

            aspect_tagger = torch.log(aspect_tagger + 1e-20)  #將屬性預測部分經過log
            logits = torch.log(logits + 1e-20)  #將情感預測部分經過log
            
            loss_fct = NLLLoss()  #因為已經過softmax和log，不能使用CrossEntropyLoss
            
            if attention_mask is not None: 
                active_loss = attention_mask.view(-1) == 1  
                active_logits = aspect_tagger.view(-1, 3)
                active_labels = torch.where(
                    active_loss, aspect_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                aspect_loss = loss_fct(active_logits, active_labels)
                
                
                active_loss = attention_mask.view(-1) == 1  
                active_logits = logits.view(-1, 4)
                active_labels = torch.where(
                    active_loss, sentiment_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                sentiment_loss = loss_fct(active_logits, active_labels)

                
                loss = aspect_loss + sentiment_loss   
                
            else:
                aspect_loss = loss_fct(aspect_tagger.view(-1, 3), aspect_labels.view(-1))
                sentiment_loss = loss_fct(logits.view(-1, 4), sentiment_labels.view(-1))
                
                loss = aspect_loss + sentiment_loss   
            
            outputs = (loss,) + (aspect_loss,) + (sentiment_loss,) + outputs  

        return outputs   


# class MyNewBertForTokenClassification(BertPreTrainedModel):
#     '''把bert輸出分別接上一層線性分類層，是屬性情感分析的模型'''
    
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#         # 將bert輸出傳到這層來預測屬性標記
#         self.aspect_classfier = nn.Linear(config.hidden_size,3)  #768 -> 3 (BIO) 
        
#         # 將bert輸出傳到這層來預測情感標記
#         self.sentiment_classifier = nn.Linear(config.hidden_size, 4) #768 -> 4 (POS/NEU/NEG/O)
        
#         self.init_weights()

        
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
                 
#         tagger_input = outputs[0]
#         tagger_input = self.dropout(tagger_input)

#         aspect_tagger = self.aspect_classfier(tagger_input)  #輸出預測的屬性標註, 維度是bsz,seq,num_labels (3,因為標BIO) 
#         logits = self.sentiment_classifier(tagger_input)   #得到最後經過情感標註後的結果
        
#         outputs = (aspect_tagger,) + (logits,) + outputs[2:]   
    
    
#         ### 輸入的label map:  {'B-NEG': 0, 'B-NEU': 1, 'B-POS': 2, 'I-NEG': 3, 'I-NEU': 4, 'I-POS': 5, 'O': 6}
#         ## 需轉成自定義的新label map : 
#             # aspect_labels : {"B" : 0, "I":1, "O": 2}
#             # sentiment_labels {"NEG": 0 ,"NEU":1, "POS":2, "O":3}
        
#         if labels is not None:
#             #處理 labels、讓它符合joint屬性模型格式
#             aspect_labels, sentiment_labels = chg_labels_to_aspect_and_sentiment(labels)
             
#             loss_fct = CrossEntropyLoss()  
#             if attention_mask is not None: 
#                 # 先算屬性標記的loss:
#                 active_loss = attention_mask.view(-1) == 1  
#                 active_logits = aspect_tagger.view(-1, 3)
#                 active_labels = torch.where(
#                     active_loss, aspect_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
#                 )
#                 aspect_loss = loss_fct(active_logits, active_labels)
                
                
#                 active_loss = attention_mask.view(-1) == 1  
#                 active_logits = logits.view(-1, 4)
#                 active_labels = torch.where(
#                     active_loss, sentiment_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
#                 )
#                 sentiment_loss = loss_fct(active_logits, active_labels)
                
#                 loss = aspect_loss + sentiment_loss  #兩個loss會一起訓練
                
#             else:
#                 aspect_loss = loss_fct(aspect_tagger.view(-1, 3), aspect_labels.view(-1))
#                 sentiment_loss = loss_fct(logits.view(-1, 4), sentiment_labels.view(-1))
                
#                 loss = aspect_loss + sentiment_loss  #兩個loss會一起訓練
            
#             outputs = (loss,) + (aspect_loss,) + (sentiment_loss,) + outputs 

#         return outputs  