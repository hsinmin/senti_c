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
'''句子情感分類的資料讀取與處理'''

import os
import copy
import csv
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentenceProcessor:
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar)) 
        
    def get_train_dev_examples(self, data_dir, mode):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, mode+".tsv")), mode)
    
    def get_test_examples_from_list(self, lines):
        '''data為list格式'''
        return self._create_examples_from_test(lines)
    
    def get_test_examples_from_tsv(self, data_dir):
        '''這裡的 data_dir 為使用者自己指定的檔名(路徑)'''
        
        ## 讀取tsv檔案
        data = self._read_tsv(data_dir) #得到一個大list，裡面每個list中包含tsv檔案中的一列資料
        
        ## 將資料整理成一個list,裡面每個元素都是一個String
        new_data = [i[0] for i in data] #因為只有文本所以取[0] 
        
        return new_data

    def get_labels(self):
        return ["0", "1"]   

    def _create_examples(self, lines, set_type):
        '''創建訓練和驗證時的資料examples'''
        
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1:]   
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _create_examples_from_test(self, lines):
        '''這是給沒有label的test資料所使用；lines型別為list'''
        
        examples = []
        for (i, line) in enumerate(lines):
            guid = "test-%s" % (i)
            text_a = line
     
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
        return examples
    
    
def sentence_convert_examples_to_features(
    examples,
    tokenizer,
    label_list=None,
    max_length=256,
    pad_token=0,
    pad_token_segment_id=0,
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
    
    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        len_examples = len(examples)
        
        if ex_index % 10000 == 0 and logging_display:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )


        labels_ids = []
        if example.label is not None: #表示並非預測時，因為預測時這個會是none
            for label in example.label: 
                labels_ids.append(float(label))
            
            
        if ex_index < 3 and logging_display:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if example.label is not None:
                logger.info("label: %s (id = %s)" % (example.label, (labels_ids,)))   

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=labels_ids
            )   
        )

    return features


def sigmoid_array(x):                                       
    return 1 / (1 + np.exp(-x))