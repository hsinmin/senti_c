# senti_c (sentiment analysis toolkit for traditional Chinese)

## 簡介
本工具為繁體中文情感分析套件，支援三種類型分析：句子情感分類、屬性術語提取、屬性情感分類；同時提供函數供使用者應用其它資料重新微調模型。

## 目錄
* [安裝方式](#安装方式)
* [功能介紹](#功能介紹)
* [範例程式](#範例程式)
* [資料](#資料)

---

## 執行環境
* python3.7 

## 安裝方式
* 要先至 https://pytorch.org/ 下載適合作業系統的PyTorch套件。

1.pip 
```bash
pip install senti_c 
```

2.from source

```bash
git clone https://github.com/julielanblue/senti_c
cd senti_c
python3 setup.py install
```

## 功能介紹
1.句子情感分類：**預測**

```bash
from senti_c import SentenceSentimentClassification

sentence_classifier = SentenceSentimentClassification()

test_data = ["我很喜歡這家店！超級無敵棒！","這個服務生很不親切..."]  
result = sentence_classifier.predict(test_data,run_split=True,aggregate_strategy=False)  # 可依據需求調整參數
```
    
* 結果如下：

![avatar](https://upload.cc/i1/2020/08/04/LsiTvH.jpg)


2.句子情感分類：**重新微調模型**

```bash
from senti_c import SentenceSentimentModel

sentence_classifier = SentenceSentimentModel()
sentence_classifier.train(data_dir="./data/sentence_data",output_dir="test_fine_tuning_sent")  # 可依據需求調整參數
```


3.屬性情感分析：**預測**

```bash
from senti_c import AspectSentimentAnalysis

aspect_classifier = AspectSentimentAnalysis()

test_data = ["我很喜歡這家店！超級無敵棒！","這個服務生很不親切..."]   
result = aspect_classifier.predict(test_data,output_result="all")  # 可依據需求調整參數
```
*  結果如下：

![avatar](https://upload.cc/i1/2020/08/04/sfOrPp.jpg)

![avatar](https://upload.cc/i1/2020/08/04/qhECn7.jpg)

![avatar](https://upload.cc/i1/2020/08/04/otg9XV.jpg)

![avatar](https://upload.cc/i1/2020/08/04/u2Exd9.jpg)



4.屬性情感分析：**重新微調模型**

```bash
from senti_c import AspectSentimentModel

aspect_classifier = AspectSentimentModel()
aspect_classifier.train(data_dir="./data/aspect_data",output_dir="test_fine_tuning_aspect")  # 可依據需求調整參數
```

## 範例程式
相關功能demo可參考examples資料夾中的function_demo檔案。



## 資料
本研究蒐集Google評論上餐廳與飯店領域評論內容、並進行句子情感分類、屬性情感分析標記 (屬性標記與情感標記)。

相關資料格式請見data資料夾。

## 引用
1.論文：    
凃育婷（2020）。基於順序遷移學習開發繁體中文情感分析工具。國立臺灣大學資訊管理學研究所碩士論文，台北市。


2.實驗室：    
Business Analytics and Economic Impact Research Lab  
Department of Information Management  
National Taiwan University     
http://www.im.ntu.edu.tw/~lu/index.htm  

## 致謝
本套件基於 Hugging Face 團隊開源的 <a href="https://github.com/huggingface/transformers">transformers</a>。 











