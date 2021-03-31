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
'''通用的(句子/屬性)一些函數'''

import random
import numpy as np
import torch
import zipfile
import requests
import os


def _download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = _get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    _save_response_content(response, destination)    

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def get_toolkit_models(model_path,types="sentence",varients="default"):
    '''用來下載與解壓縮模型'''
    
    #先下載壓縮檔 :
    #壓縮檔包含模型相關資料
    ## 程式來自 https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
    if types == "sentence":
        if varients == "default":
            file_id = '1DnSE7duvvJsqTTLWelGWGW9E8EH47ODp' 
        else:
            file_id = '1mwWLrjiYb7YyeHd5EtWKrHz_I0_YpYF-' 
    else:
        if varients == "default":
            file_id = '1HHjpR9Am9a2JgDRrqVWh6t5BRRjlqk2u' 
        else:
            file_id = '1a2RtKsGtgK6c3yCE_ABpZZZ18VcvZApv'
            
    destination = model_path+'/models.zip'
    _download_file_from_google_drive(file_id, destination)
    
    #解壓縮
    with zipfile.ZipFile(destination,'r') as zips:
        for fl_name in zips.namelist():
            zips.extract(fl_name, model_path)    
        
    #刪除壓縮檔    
    try:
        os.remove(destination)
    except:
        raise Exception("壓縮檔 %s 不存在！請重新確認！",destination)
        

def get_domain_embedding(file_path):
    '''用來下載領域詞向量檔案'''
    
    ## 程式來自 https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
    file_id = '1wVJUNhbHraehDyzo_2s5apWEoYmzRuUS'
            
    if not os.path.isfile(file_path):
        _download_file_from_google_drive(file_id, file_path)

        
def split_text_from_input_data(text):
    '''斷句程式'''   
    
    start = 0
    pos = 0
    length = len(text)  #文本長度
    sentences = []
    inQuote = False

    while (pos < length):  #一個個遍歷
        c = text[pos]

        if (c == '。' or c == '？' or c == '！' or c =='!' or c== '?' or c=='~' or c =='～'):

            if (pos >= length-1):  # 以句號、問號和感嘆號結束文本，成句
                sentence = text[start:pos+1]
                sentences.append(sentence);

                pos = pos+1
                start = pos

            else:  #句號、問號和感嘆號還有内容

                nextC = text[pos+1]

                if (nextC == '”' or nextC == '〞' or nextC =='′' or nextC == '」'): #句號、問號和感嘆號在引號内，和引號一起成句
                    sentence = text[start: pos+2]
                    sentences.append(sentence)

                    pos += 2
                    start = pos

                    inQuote = False

                elif (nextC == '。' or nextC == '？' or nextC == '！' or nextC =='!' or nextC== '?'or nextC=='~' or nextC =='～'): #多個重複的。？！
                    pos = pos + 1 

                else:
                    if (~inQuote):   #句號、問號和感嘆號不在引號内，成句
                        sentence = text[start:pos + 1]
                        sentences.append(sentence)

                        pos = pos+1
                        start = pos
                    else:  #找引號
                        pos = pos + 1

        else:
            if (c == '“' or c == '「'):
                inQuote = True
            elif (c == '”' or c == '〞' or c =='′' or c == '」'):
                inQuote = False
            pos = pos + 1


    if (pos > start):    # 還有剩餘内容，成句 (如文本沒有什麼句號之類做結尾)
        sentence = text[start:pos]
        sentences.append(sentence)

    return sentences    




