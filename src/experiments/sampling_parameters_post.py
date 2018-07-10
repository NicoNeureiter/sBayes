#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.util import load_from #, zone_similarity
import numpy as np

if __name__ == '__main__':

    TEST_SAMPLING_DIRECTORY = 'data/results/test/sampling/2018-06-13_14-40-12/sampling_e0_a1_mparticularity_0.pkl'
    samples = load_from(TEST_SAMPLING_DIRECTORY)




    auto_sim = zones_autosimilarity(samples[0], 40)
    print(auto_sim)

    # for z in range(len(zones)):
    #     intersection = np.minimum(zones[z][0], zones[z-1][0])
    #
    #     recall = np.sum(intersection)
    #     print(recall)

    #print(len(samples[0]))
    # print(len(samples[1]['true_zones_ll']))
    #print(len(samples[1]['step_likelihoods']))

    # x = [10,9,8,7,6,5,5,5,4,3,4,5,6]
    # t = 3
    #
    # def auto_corr(x, t=1):
    #     ac = np.corrcoef(np.array([x[0:len(x) - t], x[t:len(x)]]))
    #     return ac[0, 1]
    #
    # ac_x = []
    # for t in range(int(np.floor(len(x)/2))):
    #     ac_x.append(auto_corr(x, t))
    #
    # print (ac_x)