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
'''句子情感分類模型的微調/訓練'''

import glob
import json
import logging
import os,sys
import random
import copy
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import f1_score,precision_score,recall_score  

from transformers import (
    BertTokenizer,  
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    AdamW,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from .model_structure import MyNewBertForSequenceClassification  
from .utils import SentenceProcessor,sentence_convert_examples_to_features as convert_examples_to_features,sigmoid_array,split_text_from_input_data


logger = logging.getLogger(__name__)  

DEFAULT_SENTENCE_CLASSIFICATION_PRETRAIN_MODEL = 'bert-base-chinese'  


def _set_seed(seed,n_gpu):
    '''設定seed'''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

        
class SentenceSentimentModel:
    '''句子情感分類模型的訓練：使用設定的參數來訓練/微調預測模型'''  
    
    def __init__(self,
                 model_name_or_path=DEFAULT_SENTENCE_CLASSIFICATION_PRETRAIN_MODEL,
                 config_name="",
                 tokenizer_name="",
                 cache_dir="",
                 no_cuda=False,
                 local_rank=-1,
                 logging_display=True,
                 seed=42,
                ):  
        '''
        設定句子情感分類模型訓練相關參數：
        model_name_or_path : 預訓練的模型，可以為路徑或是huggingface團隊支援的模型名稱，默認為"bert-base-chinese"
        config_name : 如果不想使用預訓練模型的config，可以自己撰寫新的config檔案，這裡需放名稱或路徑，可選參數
        tokenizer_name : 如果不想使用預訓練模型的tokenizer，可指定別的名稱或路徑，可選參數
        cache_dir : 放置下載的預訓練模型的位置，可選參數
        no_cuda : 是否避免使用gpu，默認"False"
        local_rank : 是否使用平行化運算，默認"-1"
        logging_display : 設置是否顯示logging，默認為"True"
        seed : 隨機種子，默認為"42"
        '''
        
        ### 設定 cpu,gpu ###
        self.local_rank = local_rank
        
        if self.local_rank == -1 or no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            self.n_gpu = 0 if no_cuda else torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1

        ### 設定 logging ###
        self.logging_display = logging_display
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN,
        )   
        
        
        if self.logging_display:
            logger.warning(
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                self.local_rank,
                self.device,
                self.n_gpu,
                bool(self.local_rank != -1),
            )

        
        ### 設定 seed ###
        self.seed = seed
        _set_seed(self.seed,self.n_gpu)
        
        
        ### 匯入 config,tokenizer,model ###
        self.model_name_or_path = model_name_or_path
        
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier() 
        
        self.config = BertConfig.from_pretrained(
            config_name if config_name else model_name_or_path,
            num_labels=2,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model = MyNewBertForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=self.config,
            cache_dir=cache_dir if cache_dir else None,
        )

        if self.local_rank == 0:
            torch.distributed.barrier() 

        
    def _train(self,saved_args,train_dataset):  
        '''訓練模型的主要函數'''  
        
        ## 取出模型部分 ##
        model = saved_args["local_model"]  
        
        if self.local_rank in [-1, 0]:   
            tb_writer = SummaryWriter()   

        saved_args['train_batch_size'] = saved_args["per_gpu_train_batch_size"] * max(1, self.n_gpu)
        train_sampler = RandomSampler(train_dataset) if self.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=saved_args['train_batch_size'])

        if saved_args["max_steps"] > 0:
            t_total = saved_args["max_steps"]
            saved_args["num_train_epochs"] = saved_args["max_steps"] // (len(train_dataloader) // saved_args["gradient_accumulation_steps"]) + 1
        else:
            t_total = len(train_dataloader) // saved_args["gradient_accumulation_steps"] * saved_args["num_train_epochs"]

        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": saved_args["weight_decay"],
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=saved_args["learning_rate"], eps=saved_args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=saved_args["warmup_steps"], num_training_steps=t_total
        )

        
        if os.path.isfile(os.path.join(self.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.model_name_or_path, "scheduler.pt")
        ):
            optimizer.load_state_dict(torch.load(os.path.join(self.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.model_name_or_path, "scheduler.pt")))

        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        ## 平行化處理 ##
        if self.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True,
            )

        ## 以下開始訓練 ##
        if self.logging_display:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_dataset))
            logger.info("  Num Epochs = %d", saved_args["num_train_epochs"])
            logger.info("  Instantaneous batch size per GPU = %d", saved_args["per_gpu_train_batch_size"])
            logger.info(
                "  Total train batch size (w. parallel, distributed & accumulation) = %d",
                saved_args['train_batch_size']
                * saved_args["gradient_accumulation_steps"]
                * (torch.distributed.get_world_size() if self.local_rank != -1 else 1),
            )
            logger.info("  Gradient Accumulation steps = %d", saved_args["gradient_accumulation_steps"])
            logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        
        ## 檢查是否從某個檢查點繼續訓練 ##
        if os.path.exists(self.model_name_or_path):
            try:
                global_step = int(self.model_name_or_path.split("-")[-1].split("/")[0])
            except ValueError:
                global_step = 0
            epochs_trained = global_step // (len(train_dataloader) // saved_args["gradient_accumulation_steps"])
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // saved_args["gradient_accumulation_steps"])
            
            if self.logging_display:
                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0

        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(saved_args["num_train_epochs"]), desc="Epoch", disable=self.local_rank not in [-1, 0],
        )
        
        _set_seed(self.seed,self.n_gpu)  ## 放這以便重製結果
        for _ in train_iterator:  # 保留epoch的tqdm
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids" : batch[2], "labels": batch[3]}
                
                outputs = model(**inputs)
                loss = outputs[0]  

                if self.n_gpu > 1:
                    loss = loss.mean()  # 因為平行化處理
                if saved_args["gradient_accumulation_steps"] > 1:
                    loss = loss / saved_args["gradient_accumulation_steps"]

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % saved_args["gradient_accumulation_steps"] == 0 or (
                    len(epoch_iterator) <= saved_args["gradient_accumulation_steps"]
                    and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), saved_args["max_grad_norm"])

                    optimizer.step()
                    scheduler.step()  
                    model.zero_grad()
                    global_step += 1


                    if self.local_rank in [-1, 0] and saved_args["logging_steps"] > 0 and global_step % saved_args["logging_steps"] == 0 :
                        logs = {}

                        if (self.local_rank == -1 and saved_args["evaluate_during_training"]): 
                            results = self._evaluate(model,saved_args)
                            
                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value
                            
  
                        loss_scalar = (tr_loss - logging_loss) / saved_args["logging_steps"]
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["train_eval_loss"] = loss_scalar   
                        logging_loss = tr_loss
                    
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        if self.logging_display:
                            print(json.dumps({**logs, **{"step": global_step}}))

                        
                    if self.local_rank in [-1, 0] and saved_args["save_steps"] > 0 and global_step % saved_args["save_steps"] == 0:
                        ## 儲存模型 checkpoint ##
                        checkpoint_output_dir = os.path.join(saved_args["output_path"], "checkpoint-{}".format(global_step))
                        if not os.path.exists(checkpoint_output_dir):
                            os.makedirs(checkpoint_output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  
                        model_to_save.save_pretrained(checkpoint_output_dir)
                        self.tokenizer.save_pretrained(checkpoint_output_dir)

                        torch.save(saved_args, os.path.join(checkpoint_output_dir, "training_args.bin"))
                        if self.logging_display:
                            logger.info("Saving model checkpoint to %s", checkpoint_output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(checkpoint_output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(checkpoint_output_dir, "scheduler.pt"))
                        if self.logging_display:
                            logger.info("Saving optimizer and scheduler states to %s", checkpoint_output_dir)

                if saved_args["max_steps"] > 0 and global_step > saved_args["max_steps"]:
                    epoch_iterator.close()
                    break
            if saved_args["max_steps"] > 0 and global_step > saved_args["max_steps"]:
                train_iterator.close()
                break
                
        if self.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step , model , saved_args

    
    def _evaluate(self, model, saved_args, prefix=""):
        '''對模型進行驗證'''
 
        eval_dataset = self._load_and_cache_examples(saved_args, evaluate=True)   

        if not os.path.exists(saved_args["output_dir"]) and self.local_rank in [-1, 0]:
            os.makedirs(saved_args["output_dir"])

        saved_args["eval_batch_size"] = saved_args["per_gpu_eval_batch_size"] * max(1, self.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=saved_args["eval_batch_size"])

        # 多個GPU時
        if self.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        ## 以下開始對模型進行驗證 ##
        if self.logging_display:
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", saved_args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None  
        out_label_ids = None
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids" : batch[2], "labels": batch[3]}
                
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()   
                
            nb_eval_steps += 1
            
            if preds is None:
                preds = logits.detach().cpu().numpy()  # logits維度:(bsz,num-lables)
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis = 0)   #最後接起來維度會是: (總樣本數, num-labels)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        ## 評估標準 ##
        # 將 preds 的內容轉成明確預測哪一類別
        preds = sigmoid_array(preds)  #先經過sigmoid
        new_preds = np.zeros_like(preds)  #生成相同形狀和資料類型的array (這是要放轉為標籤後的,預設放0)
        new_preds [preds > saved_args["threshold"]] = 1  #將大於threshold的標籤設為1 
        
        results = {} 
        
        ### 以下為多標籤相關指標 ###
        result = {
        "loss": eval_loss,
        "multi_macro_precision" :precision_score(out_label_ids, new_preds, average="macro"),
        "multi_macro_recall" :recall_score(out_label_ids, new_preds, average="macro"),    
        "multi_macro_f1": f1_score(out_label_ids, new_preds, average="macro"),  #看 f1 score的輸入都要是 (總樣本數 * 標籤數)
        "multi_micro_precision" :precision_score(out_label_ids, new_preds, average="micro"),
        "multi_micro_recall" :recall_score(out_label_ids, new_preds, average="micro"),
        "multi_micro_f1": f1_score(out_label_ids, new_preds, average="micro")  #看 f1 score的輸入都要是 (總樣本數 * 標籤數)
        }
        results.update(result)
        
        
        ## 把值轉回類別
        y_pred = []
        for i in range(len(new_preds)):
            tmp = new_preds[i]
            
            pos = tmp[0]
            neg = tmp[1]

            if pos == 1 and neg == 0:
                y_pred.append(1)
            elif pos == 0 and neg == 1:
                y_pred.append(0)    
            elif pos == 0 and neg == 0:
                y_pred.append(2)
            else:  #衝突類別
                y_pred.append(3)
    
        y_true = []
        for i in range(len(out_label_ids)):
            tmp = out_label_ids[i]
            
            pos = tmp[0]
            neg = tmp[1]

            if pos == 1 and neg == 0:
                y_true.append(1)
            elif pos == 0 and neg == 1:
                y_true.append(0)    
            elif pos == 0 and neg == 0:
                y_true.append(2)
            else:  #衝突類別
                y_true.append(3)
        
       
        ### 以下為轉回四類別的相關指標 ###
        result = {
        "macro_precision" :precision_score(y_true, y_pred, average='macro'),
        "macro_recall" :recall_score(y_true, y_pred, average='macro'),    
        "macro_f1": f1_score(y_true, y_pred, average='macro'),  #看 f1 score的輸入都要是 (總樣本數 * 標籤數)
        "micro_precision" :precision_score(y_true, y_pred, average='micro'),
        "micro_recall" :recall_score(y_true, y_pred, average='micro'),
        "micro_f1": f1_score(y_true, y_pred, average='micro')  #看 f1 score的輸入都要是 (總樣本數 * 標籤數)
        }
        results.update(result)
        
        if self.logging_display:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))

                
        return results

    
    def _load_and_cache_examples(self, saved_args, evaluate=False): 
        '''讀取資料集&轉成特徵'''
        
        if self.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  

        processor = SentenceProcessor()
        
        mode = 'dev' if evaluate else 'train'
        cached_features_file = os.path.join(
            saved_args["data_dir"],
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(saved_args["max_seq_length"]),
            ),
        )
        
        if os.path.exists(cached_features_file) and not saved_args["overwrite_cache"]:
            if self.logging_display:
                logger.info("載入先前的 cache 檔案 %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            if self.logging_display:
                logger.info("創建 cache 檔案 %s", cached_features_file)
            
            ## 獲得 examples ##
            examples = processor.get_train_dev_examples(saved_args["data_dir"],mode) 
            labels = processor.get_labels()
            
            ## 將資料轉成特徵 ##
            features = convert_examples_to_features(
                examples,
                self.tokenizer,
                label_list=labels,
                max_length=saved_args["max_seq_length"],
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                logging_display=self.logging_display,
            )
            if self.local_rank in [-1, 0]:
                if self.logging_display:
                    logger.info("儲存 cache 檔案 %s", cached_features_file)
                torch.save(features, cached_features_file)
               
        if self.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  

        ## 將特徵轉成tensors並建成Dataset ##
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    
    def train(self,
              data_dir,
              output_dir,
              max_seq_length=256,
              threshold=0.5,
              evaluate_during_training=False,
              do_eval=True,
              eval_all_checkpoints=False,
              save_best_model=True,
              best_model_metric='micro_f1',
              per_gpu_train_batch_size=8,
              per_gpu_eval_batch_size=8,
              gradient_accumulation_steps=1,
              learning_rate=1e-5,
              weight_decay=0.0,
              adam_epsilon=1e-8,
              max_grad_norm=1.0,
              num_train_epochs=5.0,
              max_steps=-1,
              warmup_steps=0,
              logging_steps=500,
              save_steps=500,
              overwrite_output_dir=False,
              overwrite_cache=False,
             ):
        '''
        使用者可直接使用的函數，用來執行訓練模型的程式。(注意初始化class後，可傳入不同參數來重複呼叫此函數進行訓練，亦即使用同樣的預訓練模型，但用不同資料或訓練參數來進行微調！)
        參數包含：
        data_dir : 放置訓練檔案與驗證檔案的資料夾路徑，注意訓練與驗證檔案須為tsv格式
        output_dir : 放置輸出模型與相關檔案的輸出路徑
        max_seq_length : 斷字後最大的長度，默認"256"
        threshold : 將正面和負面機率轉為類別的臨界值，默認"0.5"
        evaluate_during_training : 是否在訓練過程中進行驗證，默認"False"
        do_eval : 是否在訓練結束後對模型進行驗證，默認"True"
        eval_all_checkpoints : 訓練完畢後，是否對每個檢查點進行驗證，默認為"False"
        save_best_model : 對模型/檢查點進行驗證後，是否保存驗證結果最佳的模型至output_dir中的"best_model"資料夾，默認為"True"
        best_model_metric : 判斷驗證時最佳模型的評估標準，默認為轉成四分類後的指標"micro_f1"，也可指定其他參數(指標前面加上multi表示為多標籤分類相關指標，其餘為轉成四類別後指標)："macro_precision"/ "macro_recall" / "macro_f1" / "micro_precision" / "micro_recall" / multi_macro_precision" / "multi_macro_recall" / "multi_macro_f1" / "multi_micro_precision" / "multi_micro_recall" / "multi_micro_f1"
        per_gpu_train_batch_size : 訓練時的batch size，默認為"8"
        per_gpu_eval_batch_size : 驗證時的batch size，默認為"8"
        gradient_accumulation_steps : 梯度累積，默認為"1"
        learning_rate : Adam優化器的初始學習率，默認為"1e-5"
        weight_decay : 優化器使用的weight_decay，默認為"0.0"
        adam_epsilon : Adam優化器的epsilon值，默認為"1e-8"
        max_grad_norm : 最大梯度的norm值，默認為"1.0"
        num_train_epochs : 訓練過程中的epoch數目，默認為5.0
        max_steps : 如果設為大於0數值則會覆蓋num_train_epochs，默認為"-1"
        warmup_steps : warmup的steps數目，默認為"0"
        logging_steps : 訓練過程中顯示相關logs的steps間隔數，默認為"500"
        save_steps : 訓練過程中儲存檢查點的steps間隔數，默認為"500"
        overwrite_output_dir : 是否覆寫輸出路徑中的資料，默認為"False"
        overwrite_cache : 是否覆寫放置訓練檔案與驗證檔案的資料夾路徑中的cache資料，默認為"False"
        '''
        
        if self.logging_display:
            all_vals = locals()
            del all_vals["self"]  #不須顯示此項
            logger.info("train函數的所有參數：%s",all_vals)
        
        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not overwrite_output_dir  
        ):
            raise Exception("輸出路徑 %s 中已存在其他檔案！需設置overwrite_output_dir！",output_dir)
           
        
        local_model = copy.deepcopy(self.model) #為了讓使用者可以多次呼叫同樣instance的train函數進行不同的模型訓練，將模型進行深層複製、避免訓練過程中相互改動
        local_model.to(self.device) 
        
        saved_args = locals() #取得目前變數(包含輸入的內容和前面定義的變數)，是一個dict
        
        ## 獲取訓練資料dataset ##
        train_dataset = self._load_and_cache_examples(saved_args,evaluate=False)
        
        ## 實際執行訓練 ##
        global_step, tr_loss, model, saved_args = self._train(saved_args,train_dataset)
        if self.logging_display:
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
        
        ## 將最終訓練好的模型保存至輸出路徑 ##
        if self.local_rank == -1 or torch.distributed.get_rank() == 0:
            if not os.path.exists(output_dir) and self.local_rank in [-1, 0]:
                os.makedirs(output_dir)
                
            if self.logging_display:
                logger.info("保存最終訓練好的模型到 %s", output_dir)
                
            model_to_save = (
                model.module if hasattr(model, "module") else model
            ) 
            model_to_save.save_pretrained(output_dir)   
            self.tokenizer.save_pretrained(output_dir)

            torch.save(saved_args, os.path.join(output_dir, "training_args.bin"))
        
        
        ## 對模型進行驗證 ##
        results = {}
        if do_eval and self.local_rank in [-1, 0]:
            best_name = ""  
            best_result = 0    
            
            checkpoints = [output_dir]
            if eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
            
            if self.logging_display:
                logger.info("開始驗證以下檢查點：%s", checkpoints)
            
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else "final"  
                
                model = MyNewBertForSequenceClassification.from_pretrained(checkpoint)
                model.to(self.device)
                result = self._evaluate(model,saved_args,prefix=global_step)  
                
                if result[best_model_metric] >= best_result:  
                    best_result = result[best_model_metric]
                    best_name = checkpoint
                
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
                results.update(result)
                
            ## 將驗證結果寫入檔案 ##
            output_eval_file = os.path.join(output_dir, "eval_results.txt")  
            with open(output_eval_file, "w") as writer:   
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))
            
            
            ## 將最佳的模型/檢查點保存至輸出路徑中的 "best_model" 資料夾下，使用者可於後續至指定位置取得該模型來進行預測 ##
            if save_best_model:
                if self.logging_display:
                    logger.info("保存最佳模型：%s ", best_name)
                    
                model = MyNewBertForSequenceClassification.from_pretrained(best_name)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                ) 
                best_output_dir = os.path.join(output_dir,"best_model")
                if not os.path.exists(best_output_dir):
                     os.makedirs(best_output_dir)
                model_to_save.save_pretrained(best_output_dir)
                torch.save(saved_args, os.path.join(best_output_dir, "training_args.bin"))
            
        return results   
            
 