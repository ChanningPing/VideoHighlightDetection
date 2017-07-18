# -*- coding: utf-8 -*-
from __future__ import division
import gensim.models.word2vec as w2v
from dateutil.parser import parse
import math as math
from collections import Counter
from operator import itemgetter
import pickle
import operator
import csv
import jieba
import re
import os
import pandas as pd
import sys
from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in
from sumy.summarizers.lsa import LsaSummarizer as LsaSummarizer
from sumy.summarizers.kl import KLSummarizer as KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer as SumBasicSummarizer
from sumy.summarizers.luhn import LuhnSummarizer as LuhnSummarizer

reload(sys)
sys.setdefaultencoding('utf8')

stop_words = ['啊','吗','，','的','一','不','在','人','有','是','为','以','于','上','他','而','后','之','来','及',
              '了','因','下','可','到','由','这','与','也','此','但','并','个','其','已','无','小','我','们','起',
              '最','再','今','去','好','只','又','或','很','亦','某','把','那','你','乃','它','吧','被','比','别',
              '趁','当','从','到','得','打','凡','儿','尔','该','各','给','跟','和','何','还','即','几','既','看',
              '据','距','靠','啦','了','另','么','每','们','嘛','拿','哪','那','您','凭','且','却','让','仍','啥',
              '如','若','使','谁','虽','随','同','所','她','哇','嗡','往','哪','些','向','沿','哟','用','于','咱',
              '则','怎','曾','至','致','着','诸','自',
              "按", "按照", "俺", "俺们", "阿", "别", "别人", "别处", "别是", "别的", "别管", "别说", "不", "不仅", "不但", "不光", "不单", "不只",
              "不外乎", "不如", "不妨", "不尽", "不尽然", "不得", "不怕", "不惟", "不成", "不拘", "不料", "不是", "不比", "不然", "不特", "不独", "不管",
              "不至于", "不若", "不论", "不过", "不问", "比方", "比如", "比及", "比", "本身", "本着", "本地", "本人", "本", "巴巴", "巴", "并", "并且",
              "非", "彼", "彼时", "彼此", "便于", "把", "边", "鄙人", "罢了", "被", "般的", "此间", "此次", "此时", "此外", "此处", "此地", "此", "才",
              "才能", "朝", "朝着", "从", "从此", "从而", "除非", "除此之外", "除开", "除外", "除了", "除", "诚然", "诚如", "出来", "出于", "曾", "趁着",
              "趁", "处在", "乘", "冲", "等等", "等到", "等", "第", "当着", "当然", "当地", "当", "多", "多么", "多少", "对", "对于", "对待", "对方",
              "对比", "得", "得了", "打", "打从", "的", "的确", "的话", "但", "但凡", "但是", "大家", "大", "地", "待", "都", "到", "叮咚", "而言",
              "而是", "而已", "而外", "而后", "而况", "而且", "而", "尔尔", "尔后", "尔", "二来", "非独", "非特", "非徒", "非但", "否则", "反过来说",
              "反过来", "反而", "反之", "分别", "凡是", "凡", "个", "个别", "固然", "故", "故此", "故而", "果然", "果真", "各", "各个", "各位", "各种",
              "各自", "关于具体地说", "归齐", "归", "根据", "管", "赶", "跟", "过", "该", "给", "光是", "或者", "或曰", "或是", "或则", "或", "何",
              "何以", "何况", "何处", "何时", "还要", "还有", "还是", "还", "后者", "很", "换言之", "换句话说", "好", "后", "和", "即", "即令", "即使",
              "即便", "即如", "即或", "即若", "继而", "继后", "继之", "既然", "既是", "既往", "既", "尽管如此", "尽管", "尽", "就要", "就算", "就是说",
              "就是了", "就是", "就", "据", "据此", "接着", "经", "经过", "结果", "及", "及其", "及至", "加以", "加之", "例如", "介于", "几时", "几",
              "截至", "极了", "简言之", "竟而", "紧接着", "距", "较之", "较", "进而", "鉴于", "基于", "具体说来", "兼之", "借傥然", "今", "叫", "将", "可",
              "可以", "可是", "可见", "开始", "开外", "况且", "靠", "看", "来说", "来自", "来着", "来", "两者", "临", "类如", "论", "赖以", "连",
              "连同", "离", "莫若", "莫如", "莫不然", "假使", "假如", "假若", "某", "某个", "某些", "某某", "漫说", "没奈何", "每当", "每", "慢说", "冒",
              "哪个", "哪些", "哪儿", "哪天", "哪年", "哪怕", "哪样", "哪边", "哪里", "那里", "那边", "那般", "那样", "那时", "那儿", "那会儿", "那些",
              "那么样", "那么些", "那么", "那个", "那", "乃", "乃至", "乃至于", "宁肯", "宁愿", "宁可", "宁", "能", "能否", "你", "你们", "您", "拿",
              "难道说", "内", "哪", "凭借", "凭", "旁人", "譬如", "譬喻", "且", "且不说", "且说", "其", "其一", "其中", "其二", "其他", "其余", "其它",
              "其次", "前后", "前此", "前者", "起见", "起", "全部", "全体", "恰恰相反", "岂但", "却", "去", "若非", "若果", "若是", "若夫", "若", "另",
              "另一方面", "另外", "另悉", "如若", "如此", "如果", "如是", "如同", "如其", "如何", "如下", "如上所述", "如上", "如", "然则", "然后", "然而",
              "任", "任何", "任凭", "仍", "仍旧", "人家", "人们", "人", "让", "甚至于", "甚至", "甚而", "甚或", "甚么", "甚且", "什么", "什么样", "上",
              "上下", "虽说", "虽然", "虽则", "虽", "孰知", "孰料", "始而", "所", "所以", "所在", "所幸", "所有", "是", "是以", "是的", "设使", "设或",
              "设若", "谁", "谁人", "谁料", "谁知", "随着", "随时", "随后", "随", "顺着", "顺", "受到", "使得", "使", "似的", "尚且", "庶几", "庶乎",
              "时候", "省得", "说来", "首先", "倘", "倘使", "倘或", "倘然", "倘若", "同", "同时", "他", "他人", "他们们", "她们", "她", "它们", "它",
              "替代", "替", "通过", "腾", "这里", "这边", "这般", "这次", "这样", "这时", "这就是说", "这儿", "这会儿", "这些", "这么点儿", "这么样", "这么些",
              "这么", "这个", "这一来", "这", "正是", "正巧", "正如", "正值", "万一", "为", "为了", "为什么", "为何", "为止", "为此", "为着", "无论",
              "无宁", "无", "我们", "我", "往", "望", "惟其", "唯有", "下", "向着", "向使", "向", "先不先", "相对而言", "许多", "像", "小", "些", "一",
              "一些", "一何", "一切", "一则", "一方面", "一旦", "一来", "一样", "一般", "一转眼", "由此可见", "由此", "由是", "由于", "由", "用来", "因而",
              "因着", "因此", "因了", "因为", "因", "要是", "要么", "要不然", "要不是", "要不", "要", "与", "与其", "与其说", "与否", "与此同时", "以",
              "以上", "以为", "以便", "以免", "以及", "以故", "以期", "以来", "以至", "以至于", "以致", "己", "已", "已矣", "有", "有些", "有关", "有及",
              "有时", "有的", "沿", "沿着", "于", "于是", "于是乎", "云云", "云尔", "依照", "依据", "依", "余外", "也罢", "也好", "也", "又及", "又",
              "抑或", "犹自", "犹且", "用", "越是", "只当", "只怕", "只是", "只有", "只消", "只要", "只限", "再", "再其次", "再则", "再有", "再者",
              "再者说", "再说", "自身", "自打", "自己", "自家", "自后", "自各儿", "自从", "自个儿", "自", "怎样", "怎奈", "怎么样", "怎么办", "怎么", "怎",
              "至若", "至今", "至于", "至", "纵然", "纵使", "纵令", "纵", "之", "之一", "之所以", "之类", "着呢", "着", "眨眼", "总而言之", "总的说来",
              "总的来说", "总的来看", "总之", "在于", "在下", "在", "诸", "诸位", "诸如", "咱们", "咱", "作为", "只", "最", "照着", "照", "直到",
              "综上所述", "贼死", "逐步", "遵照", "遵循", "针对", "致", "者", "则甚", "则", "咳", "哇", "哈",  "哉", "哎", "哗",
              "哟", "哦", "哩", "矣哉", "矣乎", "矣", "焉", "毋宁", "欤",  "嘻", "嘛", "嘘", "嘎登", "嘎", "嗳", "嗯", "嗬", "嗡嗡",
              "嗡", "喽", "喔唷", "喏", "喂", "啷当", "啪达", "啦", "啥", "啐", "啊", "唉", "哼唷", "哼", "咧", "咦", "咚", "咋", "呼哧", "呸",
              "呵", "呢", "呜呼", "呜", "呗", "呕", "呃", "呀", "吱", "吧哒", "吧", "吗", "吓", "兮", "儿", "亦", "了", "乎"]


def read_word_embeddings():# read word embedding
    '''
    :return: the trained word vectors
    '''
    danmu2vec = w2v.Word2Vec.load(os.path.join("trained", "danmu2vec.w2v"))

    return danmu2vec

def read_danmu(file_name,trail_start,trail_end):# read danmu data, sorted by elapsed_time
    danmu = pd.read_csv(file_name, sep=',')
    print(file_name)
    danmu = danmu.sort_values(['elapse_time'], ascending=[1])


    for index, row in danmu.iterrows():
        danmu.set_value(index, 'text', row['text'].replace('.', '').replace(' ', ''))
        #print(danmu.iloc[index])
        #print(str(index)+','+str(row['elapse_time']) + ',' + str(row['text']))

    real_end = danmu.elapse_time<=(danmu.iloc[-1]['elapse_time']-trail_end)
    danmu = danmu[ (danmu.elapse_time >=trail_start) & real_end]
    #print(danmu.head(n=50))

    #return danmu.head(n=500)
    danmu.drop_duplicates
    danmu = danmu.reset_index(drop=True)  # update index
    return danmu



def is_date(string): # test if a string is a date
    if len(string)>8: return False
    if len(string)<=8 and len(string)>=4:
        try:
            parse(string[0:4])
            return True and int(string[0:4])<2018
        except ValueError:
            return False
    else:
        return False

def generate_simplified_danmu_without_lag_calibration(danmu):
    count = 0
    simplified_danmu = []
    for index, row in danmu.iterrows():

        # print('s-' + str(count) + '['+str(row['elapse_time']) +']'+ row['text'])
        words = jieba.cut(row['text'])  # cut comment into words

        words = list(set(words))
        words = [re.sub(r'(.)\1+', r'\1\1', w) for w in words]  # handle 23333, 6666
        words = [re.sub(r'(哈)\1+', r'\1\1', w) for w in words]  # handle repetition
        words = [re.sub(r'(啊)\1+', r'\1\1', w) for w in words]  # handle repetition
        words = [re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）:：；《）《》“”()»〔〕-]+", "", w.decode("utf8")) for w
                 in words]

        current_time = float(row['elapse_time'])
        word_list = []

        for w in words:
            if not w in stop_words and w:
                if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)): continue
                word_list.append(w)
            simplified_danmu.append([index, current_time, word_list])
            count = count + 1
        return simplified_danmu

def constuct_lexical_chains(danmu,danmu2vec,max_silence,top_n, min_overlap,filename):
    '''
    :param danmu: raw danmu data in pandas frame
    :param danmu2vec: word embeddings pre-trained
    :param max_silence: threshold to hold next comment in the same chain, in seconds
    :param top_n: top n most similar words
    :param min_overlap: minimum percentage of overlap
    :return: a dictionary, key: concept; value: list of lists of consecutive lexical chains
    '''
    count = 0
    simplified_danmu = []
    for index, row in danmu.iterrows():

        #print('s-' + str(count) + '['+str(row['elapse_time']) +']'+ row['text'])
        words = jieba.cut(row['text']) # cut comment into words

        words = list(set(words))
        words = [re.sub(r'(.)\1+', r'\1\1', w) for w in words] # handle 23333, 6666
        words = [re.sub(r'(哈)\1+', r'\1\1', w) for w in words] # handle repetition
        words = [re.sub(r'(啊)\1+', r'\1\1', w) for w in words] # handle repetition
        words = [re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）:：；《）《》“”()»〔〕-]+", "", w.decode("utf8")) for w in words]

        current_time = float(row['elapse_time'])
        word_list = []

        for w in words:
            if not w in stop_words and w:
                if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)): continue

                word_list.append(w) # used for simplified_danmu

                if w in concept_dict: # if w in concept clusters
                    concept = concept_dict[w]
                    previous_time = danmu.iloc[lexical_chain_dict[concept][-1][-1][1]]['elapse_time']
                    if current_time - previous_time <= max_silence :

                        if  current_time >previous_time:
                            lexical_chain_dict[concept][-1].append((w,count))
                        if current_time == previous_time:
                            if lexical_chain_dict[concept][-1][-1][1] < index:
                                lexical_chain_dict[concept][-1].append((w, count))
                    else:
                        lexical_chain_dict[concept].append([(w,count)])

                else:

                    if not w in danmu2vec.wv.vocab: # if w is not in word embedding vocabulary

                        concept_dict[w] = w
                        lexical_chain_dict[w]=[[(w,count)]]
                    else:

                        overlap_dict = {}  # key: concept, value: a list of corresponding words
                        similar_words = danmu2vec.most_similar(w.decode('utf-8'), topn=top_n)
                        for word, similarity in similar_words:

                            if word in concept_dict:
                                if concept_dict[word] in overlap_dict:

                                    overlap_dict[concept_dict[word]].append(word)
                                else:

                                    overlap_dict[concept_dict[word]] =[word]

                            else: # a new temporal list not sharing anything with existing

                                if w in overlap_dict:

                                    overlap_dict[w].append(word)
                                else:

                                    overlap_dict[w]=[word]

                        enough_overlap = 0
                        for key,value in overlap_dict.iteritems():

                            if len(value) / top_n >= min_overlap and key!=w: # if overlap enough, merge into existing concept

                                enough_overlap = 1
                                for word in value:
                                    concept_dict[word]=key

                                if w in overlap_dict:
                                    for v in overlap_dict[w]:

                                        concept_dict[v] = key
                                concept_dict[w] = key


                                previous_time = danmu.iloc[lexical_chain_dict[key][-1][-1][1]]['elapse_time']
                                if current_time - previous_time <= max_silence :
                                    if current_time > previous_time:
                                        lexical_chain_dict[key][-1].append((w,count))
                                    if current_time == previous_time:
                                        if lexical_chain_dict[key][-1][-1][1]<index:
                                            lexical_chain_dict[key][-1].append((w, count))
                                else:
                                    lexical_chain_dict[key].append([(w,count)])
                                break

                        if enough_overlap ==0: # otherwise build a new concept

                            if w in overlap_dict:
                                for v in overlap_dict[w]:

                                    concept_dict[v]=w
                            concept_dict[w] = w
                            lexical_chain_dict[w] =[[(w,count)]]
                        #if 's' in concept_dict: print(concept_dict['s'])
        simplified_danmu.append([index, current_time,word_list])
        count = count + 1

    '''
    frequencies = []  # used for default frequency of words not in word2vec vocabulary
    for key, value in lexical_chain_dict.iteritems():
        if key in danmu2vec.wv.vocab:
            frequencies.append(danmu2vec.wv.vocab[key.decode('utf-8')].count)
    avg_frequency = sum(frequencies) / len(frequencies)
    for key, value in lexical_chain_dict.iteritems():
        frequency = -1
        if key in danmu2vec.wv.vocab:
            frequency = danmu2vec.wv.vocab[key.decode('utf-8')].count
            frequency = math.log(frequency)
        else:
            frequency = math.log(avg_frequency)
        print(key.encode('utf-8')+'['+str(frequency)+']')
        for v in value:
            for word, s_id in v:
                print(word.encode('utf-8') + '[' + str(s_id) + ']')


    for s_d in simplified_danmu:
        print(str(s_d[0])+','+(' ').join([w.encode('utf-8') for w in s_d[1]]))
    '''
    print('**************** save the concept dict')
    save_obj(concept_dict, filename+'_concept_dict')
#    print(concept_dict['s'])
    #print(simplified_danmu)

    return simplified_danmu

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def similar_words(word,danmu2vec):
    similar_words = danmu2vec.most_similar(word.decode('utf-8'), topn=10)
    for word, similarity in similar_words:
        print(word)
        print(similarity)
        print(danmu2vec.wv.vocab[word.decode('utf-8')].count)

def get_average_word_frequency(danmu2vec):
    # first get default frequency for a word not in vocabulary
    frequencies = []  # used for default frequency of words not in word2vec vocabulary
    for key, value in lexical_chain_dict.iteritems():
        if key in danmu2vec.wv.vocab:
            frequencies.append(danmu2vec.wv.vocab[key.decode('utf-8')].count)
    avg_frequency = sum(frequencies) / len(frequencies)
    return avg_frequency


def align_comments(simplified_danmu,danmu2vec,scene_length,avg_frequency):
    # re-align
    for index, row in enumerate(simplified_danmu): # each comment
        #print(index)
        if row[2]:
            chain_score = {} # several chains, each with a score of importance
            for word in row[2]:
                concept = concept_dict[word] # find concept of each word
                chains = lexical_chain_dict[concept] # find corresponding chains of the concept
                found = 0
                found_chain = []
                for chain in chains: #  scan each chain
                    for w,s_id in chain: # each (word, sentence_id) pair in a chain
                        if index==s_id: # if this is the chain where the comment is at
                            found = 1
                            found_chain = chain
                            break
                    if found==1: break
                if found == 1:
                    score = 0
                    for w, s_id in found_chain:  # add up accumlative tf*idf
                        if w in danmu2vec.wv.vocab:
                            score = score + 1 / math.log(danmu2vec.wv.vocab[w.decode('utf-8')].count)
                        else:
                            score = score + 1 / math.log(avg_frequency)
                    chain_score[concept] = (score,found_chain)

            max_concept = ''
            max_score = 0
            for key, value in chain_score.iteritems():

                if value[0] > max_score:
                    max_concept = key
                    max_score = value[0]

            found_chain = chain_score[max_concept][1] # retrieve the max found_chain
            max_s_id = found_chain[0][1] # retrieve the head(0) sentence_id(1) of the max found_chain
            start_time = danmu.iloc[max_s_id]['elapse_time']
            row[1] = start_time # modifiy time of the comment


    simplified_danmu.sort(key=lambda x: x[1])
    #for s_d in simplified_danmu:
        #print('[s-'+str(s_d[0]) + ']'+str(s_d[1])+','+(' ').join([w.encode('utf-8') for w in s_d[2]]))



    #print(simplified_danmu)

    return simplified_danmu

    # TODO: calculate intensity

def segment_danmu_to_scenes(scene_length,simplified_danmu):
    scenes = []
    current_time = scene_length
    scene =[]
    for row in simplified_danmu:
        if row[1] <= current_time:
            scene.append(row)
        else:
            scenes.append(scene)
            scene = []
            current_time = current_time + scene_length
    #for scene in scenes:
        #print('***************')
        #for row in scene:
            #print('[s-' + str(row[0]) + ']' + str(row[1]) + ',' + (' ').join([w.encode('utf-8') for w in row[2]]))
    #print(scenes)
    return scenes

def read_emotion_lexicon():
    emotion_lexicon = {'happy': [], 'surprise': [], 'fear': [], 'sad': [], 'anger': []}
    with open("data/manual corrected expanded 300 each.txt", "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            #print(line[0])
            if line[1]=='1':
                emotion_lexicon['anger'].append(line[0])
            elif line[1]=='2':
                emotion_lexicon['surprise'].append(line[0])
            elif line[1]=='3':
                emotion_lexicon['fear'].append(line[0])
            elif line[1]=='4':
                emotion_lexicon['sad'].append(line[0])
            elif line[1]=='5':
                emotion_lexicon['happy'].append(line[0])
    #print(emotion_lexicon)
    return emotion_lexicon

def calculate_emotion_scores(scenes,emotion_lexicon ):
    emotion_scores = []
    for index, scene in enumerate(scenes):
        emotion_score = {'happy': 1, 'surprise': 1, 'fear': 1, 'sad': 1, 'anger': 1}
        #print('*************** scene-'+str(index))
        for row in scene:
            #print('[s-' + str(row[0]) + ']' + str(row[1]) + ',' + (' ').join([w.encode('utf-8') for w in row[2]]))

            for w in row[2]:
                sentence_emotion_score = {'happy': 0, 'surprise': 0, 'fear': 0, 'sad': 0, 'anger': 0}
                if w in emotion_lexicon['happy']:
                    sentence_emotion_score['happy'] = 1
                elif w in emotion_lexicon['surprise']:
                    sentence_emotion_score['surprise'] = 1
                elif w in emotion_lexicon['fear']:
                    sentence_emotion_score['fear'] = 1
                elif w in emotion_lexicon['sad']:
                    sentence_emotion_score['sad'] = 1
                elif w in emotion_lexicon['anger']:
                    sentence_emotion_score['anger'] = 1
                emotion_score = Counter(emotion_score) + Counter(sentence_emotion_score)
        ##print(emotion_score)
        sum_score = emotion_score['happy'] + emotion_score['surprise'] + emotion_score['fear']+ emotion_score['sad']+emotion_score['anger']
        entropy = 0
        max_value = 0
        #print(emotion_score.values())

        for key, value in emotion_score.iteritems():

            if value > 0:
                p = value / sum_score
                entropy = entropy - p * math.log(p)
            if value > max_value:
                max_value = value
        #score = math.log(max_value) / entropy

        #score = 1 / entropy
        score =1 / entropy
        emotion_scores.append( score)
    return emotion_scores


def calculate_topic_coherence(scenes,avg_frequency,emotion_lexicon):
    print('avg frequency=' + str(avg_frequency))
    all_concept_chains = []
    topic_scores = []
    for index,scene in enumerate(scenes):
        #print('*************** scene-' + str(index))
        concept_vector = {}
        for row in scene:
            if row[2]:
                for w in row[2]:
                    if not w in emotion_lexicon['anger'] and not w in emotion_lexicon['sad'] and not w in emotion_lexicon['fear'] and not w in emotion_lexicon['surprise'] and not w in emotion_lexicon['happy']:
                        concept = concept_dict[w]
                        if concept in concept_vector:
                            concept_vector[concept].append(row[0]) # record the sentence id where each concept occurs
                        else:
                            concept_vector[concept] = [row[0]]
        all_concept_chains.append(concept_vector)
        sum_concept_num = 0
        sum_concept_score = 0


        for key, value in concept_vector.iteritems(): # calculate sum number of concepts, sum scores of concepts
            concept_num =len(list(set(value)))
            sum_concept_num = sum_concept_num + concept_num
            idf = 0
            if key.decode('utf-8') in danmu2vec.wv.vocab:
                idf = math.log(danmu2vec.wv.vocab[key.decode('utf-8')].count)
            else:
                idf = math.log(avg_frequency)
            concept_score = concept_num / idf
            sum_concept_score = sum_concept_score + concept_score


        entropy_num = 0.0  # calculate scene entropy
        entropy_score = 0.0
        max_value = 0.0
        for key, value in concept_vector.iteritems(): # calculate entropy of concepts
            concept_num = len(list(set(value)))
            idf = 0
            if key.decode('utf-8') in danmu2vec.wv.vocab:
                idf = math.log(danmu2vec.wv.vocab[key.decode('utf-8')].count)
            else:
                idf = math.log(avg_frequency)
            concept_score = concept_num / idf
            p_num = concept_num / sum_concept_num
            p_score = concept_score / sum_concept_score
            entropy_num = entropy_num - p_num * math.log(p_num)

            entropy_score = entropy_score-p_score * math.log(p_score)

            if concept_num > max_value:
                max_value = concept_num

        #print('[entropy by number]=' + str(entropy_num) + '[entropy by idf]=' + str(entropy_score))
        if entropy_num>0:
            #score_num = math.log(max_value) / (entropy_num / len(concept_vector))  # calculate score of scene based on concept distribution
            score_num = math.log(max_value) / (entropy_num )
        else:
            score_num = 0
        if entropy_score>0:
            #score_score = math.log(max_value) / (entropy_score / len(concept_vector)) # calculate score of scene based on weighted concept distribution
            #score_score = math.log(max_value) / (entropy_score )  # calculate score of scene based on weighted concept distribution
            score_score = 1/ (entropy_score)
        else:
            score_score = 0
        #print( '[score by number]=' + str(score_num)+'[score by idf]=' + str(score_score))
        topic_scores.append(score_score)

    return topic_scores,all_concept_chains



def generate_highlights(file_name,scenes,emotion_scores, topic_scores, all_concept_chains,w1, scene_length, num_highlights,danmu,compression_rate,avg_frequency):



    #num_highlights = math.ceil(highlights_length / scene_length) # how many scenes needed for highlights
    content_alpha = 0
    max_emotion_score = max(emotion_scores)
    max_topic_score = max(topic_scores)
    scene_utilities = []
    scene_lengths = [len(scene) for scene in scenes]
    for index, emotion_score in enumerate(emotion_scores):
        local_utility = w1*emotion_score/max_emotion_score + (1-w1)*topic_scores[index]/max_topic_score
        #local_utility = topic_scores[index] / max_topic_score # only topic


        # a test feature to consider the entire burst of scene after re-alighment
        #local_utility = (local_utility+1) * len(scenes[index])
        #local_utility = local_utility + len(scenes[index])
        #local_utility = content_alpha*local_utility + (1-content_alpha)*len(scenes[index])/max(scene_lengths)

        if len(scenes[index])>0:
            #local_utility = local_utility * math.log(len(scenes[index]))
            local_utility = math.log(len(scenes[index])) # only spike
        else:
            local_utility = 0
        scene_utilities.append([index, local_utility])

    scene_utilities = sorted(scene_utilities, key=itemgetter(1), reverse=True)
    print('the number of highlighted scenes = ' + str(num_highlights))
    highlights = []


    for index, scene in enumerate(scene_utilities):
        if index == num_highlights: break
        print('*********************************************')
        start_time= int(scene[0])*scene_length
        end_time = int(scene[0])*scene_length + (scene_length)
        highlights.append([start_time,end_time])

        # TODO: write highlight scene text to file, words already tokenized
        with open('data/text_summary/' + file_name + '_scene_' + str(scene[0]) + '.txt', 'wb') as file:
            for row in scenes[scene[0]]:
                m, s = divmod(row[1], 60)
                h, m = divmod(m, 60)
                print('[s-' + str(row[0]) + ']' + "%d:%02d:%02d" % (h, m, s) + ',' + (' ').join(
                    [w.encode('utf-8') for w in row[2]]))
                text = danmu.iloc[row[0]]['text']
                words = list(jieba.cut(text) ) # cut comment into words
                sorted_words = sorted(words)
                contain_date = False
                for w in sorted_words:
                    if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)):
                        contain_date = True
                        break
                if not contain_date:
                    file.write((' ').join([w.encode('utf-8') for w in words])+'.\n')
            print(scene)
        file.close()








        # now print the concept chain info
        #for key, value in all_concept_chains[scene[0]].iteritems():
            #print(key.encode('utf-8') + (' ').join([str(s_id) for s_id in value]))

        # our method summary
        #generate_scene_summary(scene, danmu,compression_rate,avg_frequency)

        # benchmark summary
        '''

        sentences_string = ''
        for row in scenes[scene[0]]:
            if row[2]:
                sentences_string = sentences_string + (' ').join([w.encode('utf-8') for w in row[2]]) + '.'
        print(sentences_string)
        #summary_benchmarks(sentences_string)
        '''


    print(highlights)

    # write candidate highlights to file
    with open('data/candidate_summary/' + file_name, 'wb') as file:
        for h in highlights:
            file.write(str(h[0]) + ',' + str(h[1]))
            file.write('\n')
    return highlights

def generate_scene_summary(scene_utility,danmu,compression_rate,avg_frequency):


    # calculate the concept distributions of each scene
    concept_vector = {}
    concept_sentence = {}
    for row in scenes[scene_utility[0]]: # for each sentence in the scene
        print(danmu.iloc[row[0]]['text'].encode('utf-8'))

        words = jieba.cut(danmu.iloc[row[0]]['text'])  # cut comment into words

        words = list(set(words))
        words = [re.sub(r'(.)\1+', r'\1\1', w) for w in words]  # handle 23333, 6666
        words = [re.sub(r'(哈)\1+', r'\1\1', w) for w in words]  # handle repetition
        words = [re.sub(r'(啊)\1+', r'\1\1', w) for w in words]  # handle repetition
        words = [re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）:：；《）《》“”()»〔〕-]+", "", w.decode("utf8")) for w
                 in words]
        words = [w for w in words if not w in stop_words]

        for w in sorted(words): # for each word
            #print(w.encode('utf-8'))
            #if w=='20170404':print(str(is_date(w))+'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'+str(w.isdigit()))
            if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)):
                break

            if w:
                if w in emotion_lexicon['happy']:
                    concept = '哈哈'
                elif w in emotion_lexicon['surprise']:
                    concept = '卧槽'
                elif w in emotion_lexicon['fear']:
                    concept = '可怕'
                elif w in emotion_lexicon['sad']:
                    concept = '泪目'
                elif w in emotion_lexicon['anger']:
                    concept = '气死了'
                else:
                    concept = concept_dict[w]
                if concept in concept_vector:
                    concept_vector[concept].append(row[0]) # record the sentence id where each concept occurs
                else:
                    concept_vector[concept] = [row[0]]
                # another dict used for accessing concept by sentence_id
                if row[0] in concept_sentence: # row[0] is sentence id
                    concept_sentence[row[0]].append(concept)
                else:
                    concept_sentence[row[0]] = [concept]


    for key, value in concept_vector.iteritems(): # print concept chains
        print(key.encode('utf-8') + (' ').join([str(s_id) for s_id in value]))

    concept_importances = {}
    for key, value in concept_vector.iteritems():
        if key.decode('utf-8') in danmu2vec.wv.vocab:
            concept_idf = math.log(danmu2vec.wv.vocab[key.decode('utf-8')].count)
        else:
            concept_idf = math.log(avg_frequency)
        #concept_importance = len(list(set(value))) / concept_idf
        concept_importance = len(list(set(value)))
        concept_importances[key]=concept_importance



    # sort concept by importance


    sorted_concept_importances = sorted(concept_importances.items(), key=operator.itemgetter(1),reverse=True)



    print('----------summary---------------')

    # use previous sentence set subtract later sentence set of concept
    valid_concepts = []
    intersection = []
    for index, concept_importance in enumerate(sorted_concept_importances):
        original_vector = list(concept_vector[concept_importance[0]])
        concept_vector[concept_importance[0]] = list(set(concept_vector[concept_importance[0]])-set(intersection))
        intersection = list(set(intersection) | set(original_vector))
        if concept_vector[concept_importance[0]]:
            valid_concepts.append(concept_importance[0])




    #print(concept_sentence)
    for index,concept in enumerate(valid_concepts):
        s_ids= list(set(concept_vector[concept])) # all sentences of this concept
        sentence_scores = {}
        for s_id in s_ids: # for each sentence
            score = 0
            #print('[s_id]='+str(s_id))

            for c in list(set(concept_sentence[s_id])):
                # print(concept_sentence[s_id])
                if c in danmu2vec.wv.vocab:
                    concept_idf = math.log(danmu2vec.wv.vocab[c.decode('utf-8')].count)
                else:
                    concept_idf = math.log(avg_frequency)
                score = score + len(list(set(concept_vector[c])))*concept_idf
                #score = score + len(list(set(concept_vector[c])))

            # score is the total score excluding the key concept, and use 1 as a low estimate
            sentence_scores[s_id] = score/ len(list(set(concept_sentence[s_id])))
            #sentence_scores[s_id] = score
        sentence_scores = sorted(sentence_scores.items(), key=operator.itemgetter(1), reverse=True)
        best_s_id = sentence_scores[0][0]
        print('[' + concept.encode('utf-8') + ']['+str(best_s_id) +']'+ ']['+str(sentence_scores[0][1]) +']'+danmu.iloc[best_s_id]['text'].encode('utf-8'))
        #if index / len(valid_concepts) >= compression_rate:
        if index==3:
            break

def summary_benchmarks(sentences_string):
    '''
    :param sentences_string: all sentences as one string, has been tokenized
    :return:
    '''
    parser = PlaintextParser.from_string(sentences_string, Tokenizer("english"))
    print('=========== Basic Sum ============')
    summarizer = SumBasicSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence

    print('=========== LSA ============')
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence

    print('===========LexRank============')
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence

    print('===========KL Divergence============')
    summarizer = KLSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence

    print('===========Luhn============')
    summarizer = LuhnSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence


if __name__ == "__main__":
    scene_length = 15  # scene length in seconds
    #num_highlights = [326,429,1130,329,813,617,747,700,642,633]
    num_highlights = [33,33, 19, 33, 19, 20, 17, 17, 32, 22, 22]
    w1 = 0.9 # weight of emotion objective
    trail_start = 100 # length of begin and end to be excluded
    trail_end = 100
    count = 0
    danmu2vec = read_word_embeddings()  # read word embedding
    emotion_lexicon = read_emotion_lexicon()  # read emotion lexicon
    compression_rate = 0.2
    total_num_comments = 0
    for file_name in os.listdir('data/danmu/'):
        lexical_chain_dict = {}
        concept_dict = {}
        avg_frequency = 0
        num_highlight = num_highlights[count]  # required highlight length in seconds
        #file_name = 'zhong guo he huo ren'
        danmu = read_danmu('data/danmu/' + file_name ,trail_start,trail_end) # read danmu
        total_num_comments = total_num_comments + danmu.shape[0]
        # danmu,danmu2vec,max_silence,top_n, min_overlap
        simplified_danmu = constuct_lexical_chains(danmu, danmu2vec, 11, 15, 0.5,file_name) # 7,15,0.5 current optimal
        avg_frequency = get_average_word_frequency(danmu2vec)
        simplified_danmu = align_comments(simplified_danmu, danmu2vec, scene_length,avg_frequency) # align danmu based on lexical chain
        scenes = segment_danmu_to_scenes(scene_length, simplified_danmu) # segment re-aligned danmu into scenes
        emotion_scores = calculate_emotion_scores(scenes, emotion_lexicon)
        topic_scores,all_concept_chains = calculate_topic_coherence(scenes,avg_frequency, emotion_lexicon)
        generate_highlights(file_name,scenes,emotion_scores, topic_scores, all_concept_chains,w1, scene_length, num_highlight,danmu,compression_rate,avg_frequency)

        count += 1
    print('total number of commments=' + str(total_num_comments))


    #similar_words('奥特曼',danmu2vec)


