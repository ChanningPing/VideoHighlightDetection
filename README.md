# Video Highlight Detection with Time-Sync Comments

This project implements the following paper:

* Ping, Q., Chen, C. (2017).Video Highlights Detection and Summarization with Lag-Calibration based on Concept-Emotion Mapping of Crowd-sourced Time-Sync Comments. In EMNLP Workshop on New Frontiers in Summarization.

The slides can be found [here](https://drive.google.com/open?id=0ByRn2qS9cc0-aE5ybVZERlNWbUE):



## Datasets

There are several datasets made available in this paper, including the large-scale word-embeddings trained from danmu on [Bilibili](https://www.bilibili.com/), the 5 basic emotion danmu lexicon (happy, sad, anger, surprise, fear) built from danmu, the original danmu of the 11 videos in the paper, and the corresponding highlight golden standards.
### Word-embeddings

Please cite the paper above when you use this dataset in your work. This [word-embedding](https://drive.google.com/open?id=0ByRn2qS9cc0-eVd5UXlXc2tmQ1U) is trained from 2,108,746 time-sync comments (danmu) from Bilibili.com. It contains 15,179,132 tokens, 91,745 unique tokens, from 6,368 long videos. 
The word-embedding can be loaded in Python when gensim is installed:

```
danmu2vec = w2v.Word2Vec.load(os.path.join(word-embedding-directory, "danmu2vec.w2v"))
```
For detailed usage of the word-embedding, please refer to [gensim](https://radimrehurek.com/gensim/models/word2vec.html).

### Emotion Lexicons

Please cite the paper above when you use this dataset in your work. This [emotion lexicon](https://drive.google.com/open?id=0ByRn2qS9cc0-bVc3eUZhaHFhR3M) contains five basic emotions: happy, sad, anger, surprise, fear.
The data is in the following format:

```
哈哈,5
```
Each line represents an emotion word in Simplified Chinese, and its corresponding emotion code: 1-anger, 2-surprise, 3-fear, 4-sad and 5-happy, separated by comma. 

### Original danmu of 11 videos

Please cite the paper above when you use this dataset in your work. This [original danmu data](https://drive.google.com/open?id=0ByRn2qS9cc0-UDJ1Y2gzcm9IYlE) contains 75,653 danmu comments for 11 long videos. 
The data format is as follows:
```
10176.700195312,2015-11-20 12:20:18,71d7ebfc,1,25,16777215,0,1358260775,想知道片尾音乐的名字
```
From left to right, each column represents:
* 1-Timestamp of comment in the timeline of the video (seconds).
* 2-Timestamp of comment in real time (yyyy-mm-dd hh:mm:ss).
* 3-User id.
* 4-Display mode of comment (1-8)
* 5-Font size of comment.
* 6-Font color
* 7-Type of pooling
* 8-History id
* 9-Text of comment

### Highlight golden standard

Please cite the paper above when you use this dataset in your work. This [highlight golden standard](https://drive.google.com/open?id=0ByRn2qS9cc0-SXE5WExOQ0Q4dTg) contains golden standard for highlights of the 11 videos, constructed from mixed-clips on Bilibili.
The data format is as follows:
```
00:58:06,00:58:15,学生合影
```
From left to right, each column represents:1-start time of highlight, 2-end time of highlight, 3-manual annotation of the highlight.

## Code
Please cite the paper above when you use this code in your work. 
There are 4 code files:
* Step_1_emotion_lexicon.py is built to construct the emotion lexicon with semi-supervision iteratively.
* Step_2_embed_lexical_chain.py is built to perform concept mapping, lexical chain construction, lag-calibration, highlight detection and generate text summary in each highlight.
* Step_3_evaluate_highlights.py is built to evaluate model against benchmarks on highlight detection.
* Step_4_evaluate_summary.py is built to evaluate model against benchmarks on highlight summmarization.

