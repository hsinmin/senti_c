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
'''句子情感分類的預測相關程式'''

import logging
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset , DistributedSampler
from tqdm import tqdm
from pathlib import Path

from transformers import (
    BertTokenizer,  
)

from .model_structure import MyNewBertForSequenceClassification  
from .utils import SentenceProcessor,sentence_convert_examples_to_features as convert_examples_to_features,sigmoid_array,split_text_from_input_data,get_toolkit_models

logger = logging.getLogger(__name__) 

SENTENCE_CLASSIFICATION_MODEL = {
   'default' : '/pretrained_model/chinese_sentence_model',  #以本研究繼續預訓練模型進行微調後的最佳模型
   'open' : '/pretrained_model/chinese_sentence_model'      #以開源預訓練模型進行微調後的最佳模型
} 
  
DEFAULT_SENTENCE_CLASSIFICATION_TOKENZIER = 'bert-base-chinese' 


class SentenceSentimentClassification:
    '''句子情感分類的主要類別：使用提供的微調模型來預測給定句子的情感'''  
    
    def __init__(self,
                 model_name_or_path='default',
                 tokenizer=DEFAULT_SENTENCE_CLASSIFICATION_TOKENZIER,
                 model_cache_folder_name='original',
                 no_cuda=False,
                 local_rank=-1,
                 logging_display=True
                ):  
        '''
        設定句子情感分類相關參數：
        model_name_or_path : 用來預測句子情感的模型，可以為放置模型的路徑、或是此工具所提供的模型名稱；默認為本工具提供的由繼續預訓練模型微調完畢的模型(參數值為"default")，如果想使用由開源預訓練模型所微調出的模型，參數值設為"open"
        tokenizer : 模型使用的tokenizer名稱，默認為"bert-base-chinese" 
        model_cache_folder_name : 存放本研究所開源模型的資料夾名稱，默認名稱為"original"，表示模型下載後存放於/pretrained_model/chinese_sentence_model/original資料夾中 
        no_cuda : 是否避免使用gpu，默認"False"
        local_rank : 是否使用平行化運算，默認"-1"
        logging_display : 設置是否顯示logging，默認為"True"
        '''
        
        self.local_rank = local_rank
        
        ### 設定 logging ###
        self.logging_display = logging_display
        logging.basicConfig(
            format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt = "%m/%d/%Y %H:%M:%S",
            level = logging.INFO if self.local_rank in [-1, 0] else logging.WARN,
        )  
        
        ### 匯入 tokenizer,model ###
        hf_cache_dir = str(Path.home()) + "/senti_c/huggingface/"
        if not os.path.exists(hf_cache_dir):
            os.makedirs(hf_cache_dir)
            
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer, cache_dir = hf_cache_dir)
        
        
        if model_name_or_path in SENTENCE_CLASSIFICATION_MODEL:
            ## 下載與解壓縮預設模型並存放於與預設模型同一資料夾 ##
            model_paths = os.path.split(os.path.realpath(__file__))[0] +  SENTENCE_CLASSIFICATION_MODEL[model_name_or_path] + "/"+ model_cache_folder_name  #取得模型資料夾位置絕對路徑
            if logging_display:
                logger.info(f"Checking model at {model_paths}")
            if not os.path.exists(model_paths):
                createok = False
                try:
                    os.makedirs(model_paths)
                    createok = True
                except Exception as e:
                    logger.warning(f"Encountered error when creating folder {model_paths}: {e}")
                
                if createok == False:                    
                    # tmpdir = tempfile.TemporaryDirectory()
                    # print(f"tmpdir = {tmpdir.name}")                    
                    homedir = str(Path.home())
                    # print(f"homedir = {homedir}")  
                    model_paths = homedir + "/senti_c/" + SENTENCE_CLASSIFICATION_MODEL[model_name_or_path] + "/" + model_cache_folder_name
                    logger.info(f"Try to create model folder at temp dir: {model_paths}")
                    os.makedirs(model_paths)
                
            get_toolkit_models(model_paths,"sentence",model_name_or_path)
            self.model = MyNewBertForSequenceClassification.from_pretrained(model_paths)
        else:
            self.model = MyNewBertForSequenceClassification.from_pretrained(model_name_or_path)
        
        
        ### 設定 cpu,gpu ###
        
        if self.local_rank == -1 or no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            self.n_gpu = 0 if no_cuda else torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1
        
        self.model.to(self.device) 

        


    def _check_input_list(self,input_data_or_path):                                    
        '''檢查輸入資料(list)的格式'''
        
        data = []
        if type(input_data_or_path) != list:
            raise Exception("輸入資料型別錯誤！請重新確認！")
        elif len(input_data_or_path) == 0 :
            raise Exception("輸入資料大小為0！請重新確認！")
        else:
            data = input_data_or_path 
        return data

    
    def _check_input_tsv(self,input_data_or_path):                                    
        '''檢查輸入資料(tsv)的格式；並將tsv資料轉成list'''
        
        data = []
        
        try:
            if not os.path.isfile(input_data_or_path):
                raise Exception        
        except:
            raise Exception("輸入資料不存在或並非檔案格式！請重新確認！")
        
        
        processor = SentenceProcessor()
        data = processor.get_test_examples_from_tsv(input_data_or_path)
        return data 
    

    def _read_input_list(self,input_data,output_dir=None,overwrite_cache=True):                                 
        '''產生符合模型需求的資料格式'''
        
        ## 獲取examples ##
        processor = SentenceProcessor()
        examples = processor.get_test_examples_from_list(input_data)
        
        if output_dir is not None: #表示有指定輸出路徑
            cached_features_file = os.path.join(
                output_dir,
                "cached_test",
            )
        else: #將路徑改成當前目錄
            cached_features_file = os.path.join(
                ".",
                "cached_test",
            )
        
        if os.path.exists(cached_features_file) and not overwrite_cache:
            if self.logging_display:
                logger.info("載入先前的 cache 檔案 %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:    
            if self.logging_display:
                logger.info("創建 cache 檔案 %s", cached_features_file)
                
            ## 將資料轉成特徵 ##
            labels = processor.get_labels()
            features = convert_examples_to_features(
                examples,
                self.tokenizer,
                label_list=labels,
                max_length=256,
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                logging_display=self.logging_display,
            )  
            
            
            if self.local_rank in [-1, 0]:
                if self.logging_display:
                    logger.info("儲存 cache 檔案 %s", cached_features_file)
                torch.save(features, cached_features_file)
        
        ## 將特徵轉成tensors並建成Dataset ##
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        return dataset
            
    
    def _get_aggregate_pred_class(self,preds):
        '''
        將同一輸入文本的各句子的預測類別作整合，獲得單一預測類別
        整合原則 : 輸入文本的各句子中，若有正有負則歸類為衝突；如果全部只有一種類別則就直接歸為此類別；如果有正面/負面加上中性，則歸為正面/負面；如果有任一衝突類別則歸為衝突類別
        '''
        
        if "衝突" in preds: 
            return "衝突"
        elif ("負面" in preds) and ("正面" in preds):
            return "衝突"
        elif ("負面" in preds) and ("中性" in preds): 
            return "負面"
        elif ("正面" in preds) and ("中性" in preds):
            return "正面"
        else:  #剩下的狀況只會是所有的都是同一類別
            return preds[0] 

    
    def predict(self,
                input_data_or_path,
                input_mode="list",
                run_split=False,
                aggregate_strategy=False,
                batch_size=8,
                output_mode="return_result",
                output_dir=None,          
                threshold=0.5,
                overwrite_cache=True,
               ):
        '''
        預測句子情感的專用函數。(注意初始化class後，可傳入不同參數來重複呼叫此函數進行預測)
        參數包含：
        input_data_or_path : 輸入數據，和input_mode相關聯與對應，默認為輸入list型別  
        input_mode : 輸入模式選擇，分為"list"或"tsv"，默認"list"  => 考慮句子和屬性部分統一成txt
        run_split : 是否對每筆輸入資料執行斷句，默認"False"
        aggregate_strategy : 如果執行斷句，是否將斷句後的各句預測情感匯總(True)、或分別呈現(False)，默認"False"
        batch_size : 一次丟入多筆待預測資料到模型，默認"8"
        output_mode : 輸出模式選擇，分為"寫入檔案+回傳預測變數(write_file)"或"單純回傳預測變數(return_result)"，默認"單純回傳預測變數(return_result)"
        output_dir : 輸出檔案路徑，可選參數
        threshold : 將正面和負面機率轉為類別的臨界值，默認"0.5"
        overwrite_cache : 是否覆寫預測資料的cache檔案，默認為True，若要重複預測同樣的內容，可將此參數改為False，程式會讀取先前儲存的cache檔案
        '''

        if self.logging_display:
            all_vals = locals()
            del all_vals["self"]  #不須顯示此項
            logger.info("predict函數的所有參數： %s",all_vals)
            logger.info("開始讀取與檢查輸入數據！")
        
        if not input_data_or_path: #如果沒有任何值
            raise Exception("缺少輸入資料！請重新確認！")
        
        if input_mode == 'list':
            ### 檢查輸入資料格式 ###
            input_data = self._check_input_list(input_data_or_path)
        else:
            ### 檢查輸入資料格式，並獲得list格式的資料 ###
            input_data = self._check_input_tsv(input_data_or_path)    
            
        
        ## 依序處理每筆資料 ##
        all_data = [] #儲存所有要丟入模型的資料
        input_data_chg = []  #儲存每個句子對應的原始文本內容，這是給後面輸出時所用
        split_index = [] #儲存原本輸入文本所對應的累積句子數 (因為當輸入文本經過斷句後，可能一個輸入文本包含多個句子，為了方便後續作匯總情感，所以儲存這項) => e.g. 輸入文本有三個，斷句後分別有1,3,2個句子，則對應的 split_index 為[1,4,6]，到時要辨別的話，便是用0:1,1:4,4:6來取得
        current_counts = 0 #目前累積句子數
        
        for data in input_data: 
            ## 執行斷句 ##
            if run_split:
                sentences = split_text_from_input_data(data)  
            else:
                sentences = [data]
            
            sentences = [line for line in sentences if (len(line) > 0 and not line.isspace())] #去除整個句子中為空白的句子 
            
            all_data.extend(sentences)
            current_counts += len(sentences)
            split_index.append(current_counts)

            input_data_chg.extend([data] * len(sentences))
        
        try:
            assert len(all_data) == current_counts == len(input_data_chg)
        except:
            raise Exception("處理過程中發生錯誤！請聯繫開發者！")
        
        
        if self.logging_display:
            logger.info("原本的總輸入文本數目：%s",len(input_data))
            logger.info("處理後的總文本/句子數目：%s",current_counts)
        
        
        ## 將資料轉成模型需要的dataset ##  
        if self.logging_display:
            logger.info("開始處理輸入資料為模型需求格式！")
        test_dataset = self._read_input_list(all_data,output_dir,overwrite_cache)
         
            
        ## 實際跑模型獲得預測結果 ##
        test_batch_size = batch_size * max(1, self.n_gpu)
        test_sampler = SequentialSampler(test_dataset) if self.local_rank == -1 else DistributedSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)
        
        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)
        
        if self.logging_display:
            logger.info("***** 開始進行預測  *****")
            logger.info("  資料數量 = %d", len(test_dataset))
            logger.info("  Batch大小 = %d", test_batch_size)
        
        preds = None  #最後接起來維度會是:(總樣本數, 2)
        for batch in tqdm(test_dataloader, desc="Predicting"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1] ,"token_type_ids": batch[2]} #因為要預測，所以沒有labels

                outputs = self.model(**inputs)
                logits = outputs[0] #因為沒有label

            if preds is None:
                preds = logits.detach().cpu().numpy()  #logits維度:(bsz,num-lables)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis = 0) 
        
        if self.logging_display:
            logger.info("已完成預測！開始整理預測結果！")
                
        ## 整理預測值 ##
        preds = sigmoid_array(preds)  #經過sigmoid 
        new_preds = np.zeros_like(preds)  #生成相同形狀和資料類型的array(這是要放轉為標籤後的,預設放0)
        new_preds [preds > threshold] = 1  #將大於threshold的標籤改為1
        
        
        ## 轉成類別 ##
        toolkit_preds = []
        for i in range(len(new_preds)):
            pos = int(new_preds[i][0])
            neg = int(new_preds[i][1])

            if pos == 1 and neg == 0:
                toolkit_preds.append("正面")
            elif pos == 0 and neg == 1:
                toolkit_preds.append("負面")    
            elif pos == 0 and neg == 0:
                toolkit_preds.append("中性")
            else:  
                toolkit_preds.append("衝突")
        
        ## 整理要輸出或返回的預測結果 ##
        if run_split:
            ## 彙總情感 ##
            if aggregate_strategy: 
                split_index.insert(0,0) #最前面增加0、方便後續處理

                all_text_preds = []  #儲存每個輸入文本對應的預測類別
                for i in range(1,len(split_index)):
                    # 取得原始的一筆輸入文本的各句子情感 #
                    text_preds = toolkit_preds[split_index[i-1]:split_index[i]] 
                    result = self._get_aggregate_pred_class(text_preds) #得到綜合後的預測類別
                    all_text_preds.append(result)

                #第一欄為原始文本，第二欄為綜合後的預測結果
                pred_result = pd.DataFrame({"Inputs":input_data ,"Preds":all_text_preds})
            else:
                #第一欄為原始文本，第二欄為斷句後句子，第三欄為各句子預測結果
                pred_result = pd.DataFrame({"Inputs":input_data_chg ,"Sentences":all_data,  "Preds":toolkit_preds})
                
        else:
            #第一欄為原始文本，第二欄為對應預測結果
            pred_result = pd.DataFrame({"Inputs":input_data ,"Preds":toolkit_preds})
            
        
        ## 返回/寫入預測結果 ##    
        if output_mode == 'write_file':
            if output_dir is not None:
                output_test_predictions_file = os.path.join(output_dir,"predictions.tsv")
            else:
                output_test_predictions_file = "predictions.tsv"  #表示會放在與目前位置同一路徑
                
            if self.logging_display:
                logger.info("將預測結果寫入檔案 %s",output_test_predictions_file)
                    
            pred_result.to_csv(output_test_predictions_file,sep='\t',index=False,header=True)
        
        return pred_result  #不管哪種模式都會回傳預測結果的變數，使用者可以依據需求選擇是否使用 
        
        
        
            
            
        
          