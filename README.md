# senti_c (Traditional Chinese sentiment analysis tool based on BERT)

## Introduction
senti_c is a sentiment analysis tool constructed based on the transformer-based Bidirectional Encoder Representations from Transformers (BERT). We adopted the bert-base-chinese pre-trained model provided by [Huggingface transformer](https://huggingface.co/transformers/) implementation.

senti_c provides two functions:
1. Sentence-level sentiment classification.
2. Aspect extraction and aspect-based sentiment analysis. 

If you use senti_c, please cite our work:<br>
**Yu-Ting Tu, 2020, Developing Sentiment Analysis Toolkit for Traditional Chinese Using Sequential Transfer Learning, (Master Thesis), Retrieved from [https://hdl.handle.net/11296/er7s7w](https://hdl.handle.net/11296/er7s7w).**

There is a [vignettee written in Chinese](https://github.com/hsinmin/senti_c/blob/master/vignettee_senti_c.ipynb).  
---

## Requirements
senti_c has been tested with Python 3.7 and 3.8. It requires [transformers](https://pypi.org/project/transformers/) Version 2.11.0. Transformers, in terms, need Pytorch 1.x and tensorflow 2.2.0. You do not need GPU to use senti_c. However, using senti_c with GPU can significantly improve its speed. 

Because of the specific versions of Python packages requires by senti_c, there is a big chance that these requirements may conflict with your existing setup. To address this issue, we strongly recommend you use [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html) to provide a dedicate environemtn for senti_c. 

## Install senti_c
To install senti_c, we need to (1) Setup a Python Virtual Environment, and (2) Install senti_c in this virtual environment. 


### (1) Setup the Python Virtual Environment
You need to execute these tasks in a terminal. First switch to a working directory, say /service/redstar/senti_c by
```console
cd /service/redstar/senti_c
```

To setup a virtual environment named vm4sentic, run the following command:
```console
python3 -m venv vm4sentic
```
Next initiate the virtual environment:
```console
source vm4sentic/bin/activate
```

### (2) Install senti_c
Run this command to install senti_c:
```console
pip3 install senti_c --no-binary=wrapt,termcolor,sacremoses
```
The parameter`--no-binary=wrapt,termcolor,sacremoses` asks pip3 to install wrapt,termcolor,sacremoses without `bdist_wheel`. Alternatively, you can just run `pip3 install senti_c` if you do not mind seeing some error messages.  

## Sentence-level Sentiment Classification
Below is an example to do sentence-level sentiment classification. Create a file name `sent_pred.py`with the following content:
```python
from senti_c import SentenceSentimentClassification

sentence_classifier = SentenceSentimentClassification(logging_level = "warning")
test_data = ["我很喜歡這家店！超級無敵棒！",
             "這個服務生很不親切...",
             "這間Fridays的空間不大，座位安排略顯擁擠，尤其是有隔板的兩人桌，真的超級小。",
             "唯一印象深刻的事... 蛤蜊好大顆，大蝦毛毛蟲好吃！"]
result = sentence_classifier.predict(test_data, run_split = True, aggregate_strategy = False)
print(result.iloc[:, 1:])
```

Run this script (via: `python3 sent_pred.py`). The results are:
```console
                                   Sentences Preds
0                                   我很喜歡這家店！    正面
1                                     超級無敵棒！    正面
2                               這個服務生很不親切...    負面
3  這間Fridays的空間不大，座位安排略顯擁擠，尤其是有隔板的兩人桌，真的超級小。    負面
4                 唯一印象深刻的事... 蛤蜊好大顆，大蝦毛毛蟲好吃！    正面
```


## Aspect-based Sentiment Analysis

Below is a sample script for aspect-based sentiment analysis. Create a script named `aspect_pred.py` with the following content: 
```python
from senti_c import AspectSentimentAnalysis
aspect_classifier = AspectSentimentAnalysis(logging_level = "warning")
test_data = ["""這間Fridays的空間不大，座位安排略顯擁擠，尤其是有隔板的兩人桌，真的超級小。服務人員態度很好，只是因為客人太多，感覺人手不足，要求東西常常要等好一陣子才來。如果希望有好一點的服務品質，建議避開週末用餐時段。""", 
             """每次經過都會被台灣炒飯給吸引，決定給它一個機會踏進去嚐鮮！有點失望，因為炒飯一般般，飯糰好難吃，冷氣超冷，串燒不推薦！ 唯一印象深刻的事... 蛤蜊好大顆，大蝦毛毛蟲好吃！ 整體環境不差，服務也可以，但餐點很一般"""]
result = aspect_classifier.predict(test_data, output_result = "all")

print("Extracted aspect terms and their polarity:")
for i, aterms in enumerate(result['AspectTermAndSentimentExtraction']):
    print(f"Sentence {i}: {aterms}")

print("\n ---\nLabels for individual tokens:")
nseg = len(result['InputWords'])
# result['AspectTermTags']
for seg in range(nseg):
    print(f"\n* Sentence {seg}:")
    a1 = result['InputWords'][seg]
    a2 = result['AspectTermAndSentimentTags'][seg]
    for x1, x2 in zip(a1, a2):
        print(f"{x1}({x2}) ", end = "")

print("")

```

Run this script (via: python3 aspect_pred.py)。The results are:


```console
Extracted aspect terms and their polarity:
Sentence 0: [('空間', 'NEG'), ('座位安排', 'NEG'), ('服務人員態度', 'POS'), ('人', 'NEG'), ('服務品質', 'NEG')]
Sentence 1: [('炒飯', 'POS'), ('炒飯', 'NEG'), ('飯糰', 'NEG'), ('串燒', 'NEG'), ('蛤蜊', 'POS'), ('環境', 'POS'), ('服
務', 'POS'), ('餐點', 'NEG')]

 ---
Labels for individual tokens:

* Sentence 0:
這(O-O) 間(O-O) F(O-O) r(O-O) i(O-O) d(O-O) a(O-O) y(O-O) s(O-O) 的(O-O) 空(B-NEG) 間(I-NEG) 不(O-O) 大(O-O) ，(O-O) 座(B-NEG) 位(I-NEG) 安(I-NEG) 排(I-NEG) 略(O-O) 顯(O-O) 擁(O-O) 擠(O-O) ，(O-O) 尤(O-O) 其(O-O) 是(O-O) 有(O-O) 隔(O-O) 板(O-O) 的(O-O) 兩(O-O) 人(O-O) 桌(O-O) ，(O-O) 真(O-O) 的(O-O) 超(O-O) 級(O-O) 小(O-O) 。(O-O) 服(B-POS) 務(I-POS) 人(I-POS) 員(I-POS) 態(I-POS) 度(I-POS) 很(O-O) 好(O-O) ，(O-O) 只(O-O) 是(O-O) 因(O-O) 為(O-O) 客(O-O) 人(O-O) 太(O-O) 多(O-O) ，(O-O) 感(O-O) 覺(O-O) 人(B-NEG) 手(O-NEG) 不(O-O) 足(O-O) ，(O-O) 要(O-O) 求(O-O) 東(O-O) 西(O-O) 常(O-O) 常(O-O) 要(O-O) 等(O-O) 好(O-O) 一(O-O) 陣(O-O) 子(O-O) 才(O-O) 來(O-O) 。(O-O) 如(O-O) 果(O-O) 希(O-O) 望(O-O) 有(O-O) 好(O-O) 一(O-O) 點(O-O) 的(O-O) 服(B-NEG) 務(I-NEG) 品(I-NEG) 質(I-NEG) ，(O-O) 建(O-O) 議(O-O) 避(O-O) 開(O-O) 週(O-O) 末(O-O) 用(O-O) 餐(O-O) 時(O-O) 段(O-O) 。(O-O)
* Sentence 1:
每(O-O) 次(O-O) 經(O-O) 過(O-O) 都(O-O) 會(O-O) 被(O-O) 台(O-O) 灣(O-O) 炒(B-POS) 飯(I-POS) 給(O-O) 吸(O-O) 引(O-O) ，(O-O) 決(O-O) 定(O-O) 給(O-O) 它(O-O) 一(O-O) 個(O-O) 機(O-O) 會(O-O) 踏(O-O) 進(O-O) 去(O-O) 嚐(O-O) 鮮(O-O) ！(O-O) 有(O-O) 點(O-O) 失(O-O) 望(O-O) ，(O-O) 因(O-O) 為(O-O) 炒(B-NEG) 飯(I-NEG) 一(O-O) 般(O-O) 般(O-O) ，(O-O) 飯(B-NEG) 糰(I-NEG) 好(O-O) 難(O-O) 吃(O-O) ，(O-O) 冷(O-O) 氣(O-O) 超(O-O) 冷(O-O) ，(O-O) 串(B-NEG) 燒(I-NEG) 不(O-O) 推(O-O) 薦(O-O)
！(O-O) 唯(O-O) 一(O-O) 印(O-O) 象(O-O) 深(O-O) 刻(O-O) 的(O-O) 事(O-O) .(O-O) .(O-O) .(O-O) 蛤(B-POS) 蜊(I-POS) 好(O-O) 大(O-O) 顆(O-O) ，(O-O) 大(O-POS) 蝦(I-POS) 毛(O-O) 毛(O-O) 蟲(O-O) 好(O-O) 吃(O-O) ！(O-O) 整(O-O) 體(O-O) 環(B-POS)
境(I-POS) 不(O-O) 差(O-O) ，(O-O) 服(B-POS) 務(I-POS) 也(O-O) 可(O-O) 以(O-O) ，(O-O) 但(O-O) 餐(B-NEG) 點(I-NEG) 很(O-O) 一(O-O) 般(O-O)
```













