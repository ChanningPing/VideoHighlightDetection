# -*- coding: utf-8 -*-
from __future__ import division
import gensim.models.word2vec as w2v
import os
import sys
import csv

reload(sys)
sys.setdefaultencoding('utf8')


# manually built seeds for each emotion, by examining top , code: anger-1, surprise-2, fear-3, sad-4, happy-5, neutral-0
emotion_seeds = {
    'happy':['哈哈','笑','hh','233.233','搞笑','感动','开心','逗','撒花','恍恍惚惚','红红火火','赞','滑稽','喜感','美好','治愈','好笑'],
    'surprise':['卧槽','居然','突然','竟然','懵','莫名','妈呀','woc','好方','握草','啧啧','莫名其妙','没想到'],
    'sad': ['哭','心疼','可怜','泪目','QAQ','疼','虐','难受','呜呜','悲剧','ಥ','惨','难过','心酸','绝望','伤心','悲伤','压抑','好惨','qwq','眼泪'],
    'fear':['恐怖','吓','护体','可怕','吓死','怕','害怕','吓人','惊悚','怕怕','紧张','诡异','担心','好怕'],
    'anger':['尼玛','看不下去','受不了','活该','滚','吐槽','人渣','骂','贱','无语','摔','生气','好气','过分','自私','妈蛋','逗我','该死','废话']
}
expaned_seeds=[]
expanded_emotions=[]

def prepare_word_embeddings():

    '''
    :return: the trained word vectors
    '''
    danmu2vec = w2v.Word2Vec.load(os.path.join("trained", "danmu2vec.w2v"))
    return danmu2vec

def expand_emotion_seeds(emotion_seeds,danmu2vec,percentage,min_similarity):
    '''
    :param seeds: a dictionary of the emotional seeds
    :param danmu2vec: the trained word vectors
    :param percentage: the threshold for covering at least what percentage of the current seed set
    :param min_similarity: the threshold of minimum similarity
    :return: expanded seed set by print
    '''
    # get vocabulry size
    print('the vocabulary size is '+str(len(danmu2vec.wv.vocab)))
    # get expanded seed set
    for key, value in emotion_seeds.iteritems(): # for each type of emotion
        print('********************************' + key)
        previous_length = len(value)
        for word in value: # for each seed word
            top_similar_words = danmu2vec.most_similar(word.decode('utf-8'), topn=100)      # get most similar words
            for similar_word, similarity in top_similar_words:
                if similarity<min_similarity: break # keep only similar words exceeding threshold
                count = 0
                for shared_word in value: # compare similar word with seed words
                    if shared_word != word and danmu2vec.wv.similarity(similar_word.decode('utf-8'), shared_word.decode('utf-8'))>=min_similarity:
                        count = count+1
                        if count/len(value)>=percentage and not similar_word in value:
                            value.append(similar_word)
                            print(similar_word)
                            break
            if len(value)>300: break
    # add emotion label to the seed words
    count = 1
    emotion_labels = []
    emotion_words = []
    for key, value in emotion_seeds.iteritems():
        #print('*******************************' + key)
        for word in value:
            #print(word.encode('utf-8'))
            emotion_labels.append(count)
            emotion_words.append(word)
        count = count + 1
    #print('the raw expanded seed set:')
    #for index, value in enumerate(emotion_words):
        #print(value.encode('utf-8') + ',' + str(emotion_labels[index]))

def read_expanded_seeds():
    with open("data/manual corrected expanded 300 each.txt", "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            #print(line[0])
            if line[1]=='1':
                emotion_seeds['anger'].append(line[0])
            elif line[1]=='2':
                emotion_seeds['surprise'].append(line[0])
            elif line[1]=='3':
                emotion_seeds['fear'].append(line[0])
            elif line[1]=='4':
                emotion_seeds['sad'].append(line[0])
            elif line[1]=='5':
                emotion_seeds['happy'].append(line[0])
    print(emotion_seeds)
    return emotion_seeds



def print_top_n_words(n,danmu2vec):
    '''
    :param n: number of top words in the vocabulary by frequency
    :param danmu2vec: the trained word vectors
    :param expaned_seeds: the manually cleaned seed set
    :return: top n words in the vocabulary by frequency, excluding those in cleaned expanded seed set
    '''
    # first get the cleaned expanded seed set

    word_count_dict = {}
    for word, vocab_obj in danmu2vec.wv.vocab.items():
        word_count_dict[word]=vocab_obj.count
    import operator
    word_count_dict = sorted(word_count_dict.items(), key=operator.itemgetter(1),reverse=True)
    for word,count in word_count_dict[:n+1]:
        if not word in expaned_seeds:
            print(word.encode('utf-8') + ':'+str(count))

def train_emotion_classifier(danmu2vec):
    # read the neutral class and merge to all emotion words
    with open("data/manual neutral.txt", "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            expaned_seeds.append(line[0])
            expanded_emotions.append(0)
    # print merge result
    for index, word in enumerate(expaned_seeds):
        print(word.encode('utf-8') + ',' + str(expanded_emotions[index]))

    emotion_X = []
    emotion_Y = expanded_emotions
    emotion_words = expaned_seeds
    for index, word in enumerate(expaned_seeds):
        print(word.encode('utf-8'))
        emotion_X.append(danmu2vec.wv[word.decode('utf-8')].tolist())


    # start training
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn import linear_model
    '''

    #AdaBoost
    print('************* AdaBoost')
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, emotion_X,  emotion_Y )
    print(scores)
    clf.fit(emotion_X, emotion_Y)
    predict_Y = clf.predict(emotion_X)
    print(confusion_matrix(emotion_Y, predict_Y))

    # using Random Forrest
    print('************* random forest')
    clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split = 2, random_state = 0)
    scores = cross_val_score(clf, emotion_X, emotion_Y)
    print(scores)
    clf.fit(emotion_X, emotion_Y)
    predict_Y = clf.predict(emotion_X)
    print(confusion_matrix(emotion_Y, predict_Y))

    # using softmax
    print('************* soft max')
    clf = linear_model.LogisticRegression(C=1e5,multi_class='multinomial',solver='lbfgs')
    scores = cross_val_score(clf, emotion_X, emotion_Y)
    print(scores)
    clf.fit(emotion_X, emotion_Y)
    predict_Y = clf.predict(emotion_X)
    print(confusion_matrix(emotion_Y, predict_Y))
    '''
    # using extra trees
    print('************* extra trees')
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, emotion_X, emotion_Y)
    print(scores)
    clf.fit(emotion_X, emotion_Y)
    predict_Y = clf.predict(emotion_X)
    print(confusion_matrix(emotion_Y, predict_Y))
    print('word,neutral,anger,surprise,fear,sad,happy')
    limits = 0
    for word in danmu2vec.wv.vocab:
        test_x = danmu2vec.wv[word.decode('utf-8')].tolist()
        test_y = clf.predict_proba(test_x)
        print(word + (',').join(str(i) for i in test_y[0]))
        limits = limits + 1
        if limits>1000: break








def main():
    danmu2vec = prepare_word_embeddings()
    emotion_seeds = read_expanded_seeds()
    expand_emotion_seeds(emotion_seeds, danmu2vec, 0.05, 0.6)  #  expansion
    #print_top_n_words(5000)



main()

