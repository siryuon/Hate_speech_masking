# Data Analysis Capstone Design (2021-2, KyunhgHee Univ.)
## TOXIC sentence masking using semi-supervised learning

### Author
 * Cho Moon Gi [@siryuon](https://github.com/siryuon)

### Used data
 * https://github.com/kocohub/korean-hate-speech: labeled data, unlabeled data
 * https://github.com/2runo/Curse-detection-data: data
 * https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-007
 * Data crawled from various Internet communities (Dcinside, Fmkorea, etc...)

### Requirements
 * TensorFlow
 * scikit-learn
 * Mecab(KoNLPy)
 * gensim == 3.8.3
 * pandas
 * matplotlib
 * tweepy == 3.8.0
 * eunjeon
 * tkinter

### Directory
```
DataAnalysis_CapstoneDesign
├── Main
│   ├── ckpt
│   ├── MainProgram.py
│   └── utils.py
├── Practice
├── Reports
├── Works
│   ├── post.html
│   └── user.html # user 창
├── ckpt
├── image
└── README.md
```
### Embedding Model
 * FastText - gensim library

### Language Model (Base)
 * BiLSTM
 * RNN
 * GRU
 * Attention
 * 1D-CNN  
 
### Background
 - With the recent increase in Internet broadcasting-type SNS such as YouTube and TikTok through the spread of smartphones, it is easy for people who use smartphones and the Internet to indiscriminately be exposed to various new kinds of TOXIC WORD.
 - TOXIC WORD encountered in this way is expected to be used as anonymity in news comments, in-game chats, and anonymous communities.
 - So, if someone post such abusive language on the Internet, we need a system that automatically filters the sentence.  

### Goal
 - The goal is to create a model that can distinguish as accurately as possible whether the sentence is an abusive or non-profane sentence when presented in a sentence (short sentences of 2 to 3 lines, such as Korean, in-game chat, or Internet community comments).
 - Then, it automatically filters the TOXIC WORD by masking it with *.  

### Process
1. Text Preprocessing (Unlabel data) - Mecab(KoNLPy)
2. Make word embedding vector using FastText - FastText
3. To balancing labels, augmenting toxic data using FastText's **most_similar** method (Synonym Replacement)
4. Vectorize and padding train and test dataset - TensorFlow
5. Train models - BiLSTM, RNN, GRU, 1D-CNN, Attention, BERT, KoBERT, ETC...
6. Predict whether a given sentence is a toxic sentence
7. Masking toxic words with * by predicting the toxic probability of each word in a sentence
8. Implement program with tkinter

### Compare model performance
| **Model** | **Precision** | **Recall** | **Test Accuracy**
| ----------- | ------------ | ------------ | ------------ |
| 1DCNN    |0.83|0.96| 0.89         |
| BiLSTM    |0.91|0.91| 0.91         |
| Double-BiLSTM |0.94|0.89| 0.92   |
| Double-1DCNN |0.85|0.96| 0.89   |
| GRU |0.92|0.91| 0.92   |
| Attention+BiLSTM+GRU |0.91|0.93| 0.92   |
| BERT| 0.75|0.76|0.89|
| KoBERT|0.71|0.75|0.90 | 0.89   |
| Attention+BiLSTM+LSTM+GRU|0.86|0.96 | 0.90   |
| Deeper Attention |0.79|0.98|0.86| 0.89   |
| Node Change using best Attention| 0.82|0.97|0.88|
| Attention Refine|0.92|0.95|0.94|

(Batch size=100, epochs=20)
(* epoch 30 in Attention Refine)

### Best Model Architecture
![image](https://github.com/siryuon/DataAnalysis_CapstoneDesign/blob/081acd73fce5e5f225410adf46e37ca6ce824b53/image/architecture.png)  

### Best Model Confusion Matrix
![image](https://github.com/siryuon/DataAnalysis_CapstoneDesign/blob/928f79c2e2e03648b1d9808a6a88753a89ade5fa/image/confusion.png)  

### Masking Example
1. Normal Sentence
```
Regexed Text:          이 프로그램이 우리 계획의 시발점이다  
Tokenized Text:        [['이', '프로그램', '우리', '계획', '시발점']]
0.0% 확률로 욕설 문장입니다.
----------------------------------------
욕설 부분 분석

이	: 18.57% 확률로 욕설 부분
프로그램	: 0.02% 확률로 욕설 부분
우리	: 0.11% 확률로 욕설 부분
계획	: 0.01% 확률로 욕설 부분
시발점	: 0.08% 확률로 욕설 부분


Original Text:  이 프로그램이 우리 계획의 시발점이다. 
Masked Text:    이 프로그램이 우리 계획의 시발점이다.
```

2. Toxic Sentence
```
Regexed Text:          아 씨발 진짜 개 좆같네
Tokenized Text:        [['아', '씨발', '진짜', '개', '좆같']]
99.75% 확률로 욕설 문장입니다.
----------------------------------------
욕설 부분 분석

아	: 2.61% 확률로 욕설 부분
씨발	: 99.62% 확률로 욕설 부분
진짜	: 4.43% 확률로 욕설 부분
개	: 81.81% 확률로 욕설 부분
좆같	: 90.25% 확률로 욕설 부분


Original Text:  아 씨발 진짜 개 좆같네
Masked Text:    아 ** 진짜 * **네
```

### Program Image

1. Normal Sentence  
![image](https://github.com/siryuon/DataAnalysis_CapstoneDesign/blob/29f79ce96a53bf85cf0511e385fa31cdf0a75a85/image/Example_1.JPG)  

2. Toxic Sentence  
![image](https://github.com/siryuon/DataAnalysis_CapstoneDesign/blob/29f79ce96a53bf85cf0511e385fa31cdf0a75a85/image/Example_2.JPG)
