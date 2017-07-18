from __future__ import division
import datetime
import time
import os
import csv
import sys
import math
import pandas as pd
from random import randint
from operator import itemgetter
from textwrap import wrap
# read highlight data


def read_data(candidate_dir, reference_dir):
    time_sum = 0
    count = 0
    test_data = {}  # {movie_title: {candidate:[],reference:[]}}
    intervals = []
    for filename in os.listdir(reference_dir):
        print(filename) # print movie name
        candidate = []
        reference = []

        with open(os.path.join(reference_dir, filename)) as csvfile:  # read a candidate

            highlights = csv.reader(csvfile, delimiter=',')
            for h in highlights:
                x = time.strptime(h[0], '%H:%M:%S')
                start = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
                x = time.strptime(h[1], '%H:%M:%S')
                end = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
                print(h[0])
                print(start)
                print(h[1])
                print(end)
                intervals.append(int(end-start))
                reference.append([int(start),int(end)])
                time_sum = time_sum + int(end)-int(start)
                count += 1

                #if int(end)-int(start)<=0: print('alert!'+h[2])
        #print('=============[time_sum]='+str(time_sum))
        #print(reference)

        with open(os.path.join(candidate_dir, filename)) as csvfile:  # read a candidate
            highlights = csv.reader(csvfile, delimiter=',')
            for h in highlights:
                candidate.append([int(h[0]), int(h[1])])
                count -= 1
                if count <0:
                    break
        #print(candidate)
        pair_data = {}
        pair_data['reference'] = reference
        pair_data['candidate'] = candidate
        pair_data['time_sum'] = time_sum
        test_data[filename] = pair_data
    print('total_length_shots='+str(time_sum))
    print('total_num_shots=' + str(count))

    print('intervals=')
    print(intervals)

    from collections import Counter
    intervals = Counter(intervals)
     # Returns the highest occurring item
    print('mode of shot length=')
    print(intervals.most_common())


    return test_data

def generate_uniform_data(test_data, scene_length,movie_boundary):

    uniform_data = {}
    for key, value in test_data.iteritems():
        reference = sorted(value['reference'], key=itemgetter(0))
        candidate = []
        start = int(movie_boundary[key][0]) # start time of movie
        end = int(movie_boundary[key][1]) # end time of movie
        highlight_num = len(reference) # number of scenes in reference
        step =int(math.ceil( (end-start) / highlight_num) )# interval length for uniform sampling
        for i in range(start,end,step):
            candidate.append([i,i+scene_length])
        pair_data ={}
        pair_data['reference'] = reference
        pair_data['candidate'] = candidate
        uniform_data[key]=pair_data
    #print(uniform_data)
    return uniform_data

def generate_random_data(test_data, scene_length,movie_boundary):

    random_data = {}

    for key, value in test_data.iteritems():
        already_used_scene_id = []
        reference = sorted(value['reference'], key=itemgetter(0))
        candidate = []
        start = int(movie_boundary[key][0])  # start time of movie
        end = int(movie_boundary[key][1])  # end time of movie
        highlight_num = len(reference) # number of scenes in reference
        intervals =int(math.ceil( (end-start) / scene_length) )# interval length for uniform sampling
        while True:
            scene_id = randint(0, intervals)
            if not scene_id in already_used_scene_id:
                #print(scene_id)
                candidate.append([scene_id*scene_length,scene_id*scene_length+scene_length])
                already_used_scene_id.append(scene_id)
                if len(already_used_scene_id)==highlight_num:
                    break
        pair_data ={}
        pair_data['reference'] = reference
        pair_data['candidate'] = candidate
        random_data[key]=pair_data
    print(random_data)
    return random_data

def generate_spike_data(test_data, scene_length):
    # improve-1: remove head and tail by 1 scene

    spike_data = {}
    movie_boundary = {}
    # read in the danmu data, and trim the start and end according to reference
    for key, value in test_data.iteritems():
        reference = sorted(value['reference'], key=itemgetter(0)) # sort reference

        highlight_num = len(reference) # number of highlights required
        danmu = pd.read_csv('data/raw/' + key, sep=',')
        danmu = danmu.sort_values(['elapse_time'], ascending=[1]) # sort on elapse_time
        danmu = danmu.reset_index(drop=True)  # update index
        min_time = danmu['elapse_time'].min(axis=0)
        max_time = danmu['elapse_time'].max(axis=0)
        print('[' + str(min_time) + ',' + str(max_time) + ']')
        movie_boundary[key]=[min_time,max_time]


        # cut danmu into scenes
        scenes = [] # store [index, number of danmu] per scene
        current_time = min_time + scene_length
        count = 0
        scene_id = 0
        for index, row in danmu.iterrows():
            if row['elapse_time'] <= current_time:
                count += 1
            else:
                scenes.append([scene_id,count]) # [index,count]
                count = 0
                scene_id += 1
                current_time = current_time + scene_length
        sorted_scenes = sorted(scenes,key=itemgetter(1),reverse=True) # sorted based on count

        candidate = []
        count = 0
        for index,danmu_num in sorted_scenes:
            if index<3 or index>len(sorted_scenes)-3:
                continue
            #print(str(index)+','+str(danmu_num))
            candidate.append([index*scene_length,index*scene_length+scene_length])
            count += 1
            if count==highlight_num:
                break
        pair_data = {}
        pair_data['reference'] = reference
        pair_data['candidate'] = candidate
        spike_data[key] = pair_data
    return movie_boundary,spike_data











def calculate_ROGUE_1(test_data,transition_length):
    recalls = []
    precisions = []
    f_1s = []
    for key, value in test_data.iteritems():
        print('=============the movie is: ' + key)

        print([[start,(end-start)] for [start,end] in sorted(value['reference'], key=itemgetter(1))])
        print([[start,(end-start)] for [start,end] in sorted(value['candidate'], key=itemgetter(1))])

        # calcualte recall
        ROGUE_1 = 0
        reference = value['reference']
        candidate =  value['candidate']
        for ref in reference:
            for cand in candidate:
                ref_start = ref[0]-transition_length
                ref_end = ref[1] + transition_length
                #overlap = max(0, min(ref[1], cand[1]) - max(ref[0], cand[0]) + 1)
                overlap = max(0, min(ref_end, cand[1]) - max(ref_start, cand[0]) + 1)
                if overlap>0:
                    ROGUE_1 = ROGUE_1 + 1
                    #if overlap == (cand[1] - cand[0] + 1):
                        #candidate.remove(cand)
                    break
        print('match count='+str(ROGUE_1))
        print('reference length='+str(len( value['reference'])))
        ROGUE_1 = ROGUE_1 / len( value['reference'])
        recalls.append(ROGUE_1)
        print('[ROGUE_L]='+str(ROGUE_1))

        #calculate precision
        precision = 0
        reference = value['reference']
        candidate = value['candidate']
        for cand in candidate:
            for ref in reference:
                ref_start = ref[0] - transition_length
                ref_end = ref[1] + transition_length
                # overlap = max(0, min(ref[1], cand[1]) - max(ref[0], cand[0]) + 1)
                overlap = max(0, min(ref_end, cand[1]) - max(ref_start, cand[0]) + 1)
                if overlap > 0:
                    precision = precision + 1
                    break
                    #if overlap == (cand[1] - cand[0] + 1):
                        #candidate.remove(cand)
        print('match count=' + str(precision))
        print('candidate length=' + str(len(value['candidate'])))
        precision = precision / len(value['candidate'])
        precisions.append(precision)
        print('[precision]=' + str(precision))

        if precision+ROGUE_1:
            f_1 =2*precision*ROGUE_1/(precision+ROGUE_1)
        else:
            f_1 = 0
        f_1s.append(f_1)
        print('[F-1 measure]='+str(f_1))


        # calculate F-measure
        #F_measure = 2* ROGUE_1 * precision / (ROGUE_1 + precision)
        #print('[F_measure]=' + str(F_measure))


        # calculate average distance
        distance_sum = []
        distance_max = 0
        for ref in value['reference']:
            min_distance = sys.maxint
            for cand in value['candidate']:
                if ref[1]<cand[0]:
                    min_distance = min(min_distance,cand[0] - ref[1])
                    distance_max = max(distance_max,cand[0] - ref[1])
                elif cand[1]<ref[0]:
                    min_distance = min(min_distance, ref[0] - cand[1])
                    distance_max = max(distance_max, ref[0] - cand[1])
            distance_sum.append(min_distance)
        distance_avg = sum(distance_sum) / (len(distance_sum))
        print('[average distance]=' + str(distance_avg))



    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_f_1 = sum(f_1s) / len(f_1s)
    print('[average recall]='+str(avg_recall)+'[average precision]=' + str(avg_precision)+'[f-1 measure]='+str(avg_f_1))



def plot_overlap(test_data,spike_data,random_data, uniform_data):


    import matplotlib.pyplot as plt
    for key,value in test_data.iteritems():
        fig, ax = plt.subplots()
        fig.suptitle(key, fontsize=10)  # Add the text/suptitle to figure
        reference = [[start,(end-start)] for [start,end] in sorted(value['reference'], key=itemgetter(0))]
        candidate = [[start,(end-start)] for [start,end] in sorted(value['candidate'], key=itemgetter(0))]
        spike = [[start, (end - start)] for [start, end] in sorted(spike_data[key]['candidate'], key=itemgetter(0))]
        random = [[start, (end - start)] for [start, end] in sorted(random_data[key]['candidate'], key=itemgetter(0))]
        uniform = [[start, (end - start)] for [start, end] in sorted(uniform_data[key]['candidate'], key=itemgetter(0))]


        ax.broken_barh(reference, (0, 7), facecolors='skyblue')
        ax.broken_barh(candidate, (10, 7), facecolors=('red'))
        ax.broken_barh(spike, (20, 7), facecolors=('lightgreen'))
        ax.broken_barh(uniform, (30, 7), facecolors='orange')
        ax.broken_barh(random, (40, 7), facecolors=('purple'))


        ax.set_ylim(0, 50)
        ax.set_xlim(0, 10000)
        ax.set_xlabel('elapsed time (seconds)',fontsize=24)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.tick_params(axis='x', labelsize=18)
        labels =['Reference', 'Our method','Spike-selection','Uniform-Selectoin','Random-Selection']
        labels = ['\n'.join(wrap(l, 10)) for l in labels]
        ax.set_yticklabels(labels,fontsize=24)
        ax.grid(True)


        plt.show()


if __name__ == "__main__":
    scene_length = 27
    transition_length = 5
    candidate_dir = 'data/candidate_summary/'
    reference_dir = 'data/reference_summary/'
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^candidate')
    test_data = read_data(candidate_dir, reference_dir)
    calculate_ROGUE_1(test_data,transition_length)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^spike')
    movie_boundary,spike_data = generate_spike_data(test_data, scene_length)
    calculate_ROGUE_1(spike_data,transition_length)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^uniform')
    uniform_data = generate_uniform_data(test_data,scene_length,movie_boundary )
    calculate_ROGUE_1(uniform_data,transition_length)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^random')
    random_data = generate_random_data(test_data, scene_length,movie_boundary)
    calculate_ROGUE_1(random_data,transition_length)

    plot_overlap(test_data,spike_data,random_data,uniform_data)
