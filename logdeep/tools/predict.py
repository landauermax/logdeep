#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
import math
from collections import Counter
sys.path.append('../../')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)


def generate(name):
    window_size = 10
    logs = {}
    length = 0
    with open(name, 'r') as f:
        for ln in f.readlines():
            if ',' in ln:
                # Remove sequence identifier if available
                ln = ln.split(',')[1]
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = [-1] * window_size + ln + [-1] * (window_size + 1 - len(ln))
            logs[tuple(ln)] = logs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(logs)))
    return logs, length

def get_fone(tp, fn, tn, fp):
    # Compute the F1 score based on detected samples
    if tp + fp + fn == 0:
        return 'inf'
    return tp / (tp + 0.5 * (fp + fn))

def print_results(tp, fn, tn, fp, num_cand, det_time):
    # Compute metrics and return a dictionary with results
    if tp + fn == 0:
        tpr = "inf"
    else:
        tpr = tp / (tp + fn)
    if fp + tn == 0:
        fpr = "inf"
    else:
        fpr = fp / (fp + tn)
    if tn + fp == 0:
        tnr = "inf"
    else:
        tnr = tn / (tn + fp)
    if tp + fp == 0:
        p = "inf"
    else:
        p = tp / (tp + fp)
    fone = get_fone(tp, fn, tn, fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = "inf"
    if tp + fp != 0 and tp + fn != 0 and tn + fp != 0 and tn + fn != 0:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    print('')
    print(' Time=' + str(det_time))
    print(' Cand=' + str(num_cand))
    print(' TP=' + str(tp))
    print(' FP=' + str(fp))
    print(' TN=' + str(tn))
    print(' FN=' + str(fn))
    print(' TPR=R=' + str(tpr))
    print(' FPR=' + str(fpr))
    print(' TNR=' + str(tnr))
    print(' P=' + str(p))
    print(' F1=' + str(fone))
    print(' ACC=' + str(acc))
    print(' MCC=' + str(mcc))
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'p': p, 'f1': fone, 'acc': acc, 'threshold': num_cand, 'time': det_time}

class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate(self.data_dir + self.data_dir.split('/')[-2].split('_')[0] + '_test_normal')
        test_abnormal_loader, test_abnormal_length = generate(self.data_dir + self.data_dir.split('/')[-2].split('_')[0] + '_test_abnormal')
        TP = {}
        FP = {}
        max_num_classes = self.num_candidates
        for num_cand in set(range(1, max_num_classes)):
            TP[num_cand] = 0
            FP[num_cand] = 0
        # Test the model
        start_time = time.time()
        with torch.no_grad():
        #if False:
            for line in tqdm(test_normal_loader.keys()):
                num_cand_no_anom = set(range(1, max_num_classes))
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    if label == -1:
                        continue
                    seq1 = [0] * self.num_classes
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output, 1)[0]#[-self.num_candidates:]
                    cand_anom = set()
                    for num_cand in num_cand_no_anom:
                        if label not in predicted[-num_cand:]:
                            #print(str(label) + ' not in ' + str(predicted[-num_cand:].tolist()))
                            #if label == -1:
                            #    print(i)
                            #    print(line)
                            #    print(line[i + self.window_size])
                            #else:
                            #    print('FP: ' + str(label) + ' only predicted at pos ' + str(len(predicted.tolist()) - predicted.tolist().index(label)))
                            FP[num_cand] += test_normal_loader[line]
                            cand_anom.add(num_cand)
                        #break
                    num_cand_no_anom = num_cand_no_anom.difference(cand_anom)
                    if len(num_cand_no_anom) == 0:
                        break
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                num_cand_no_anom = set(range(1, max_num_classes))
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * self.num_classes
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output, 1)[0]#[-self.num_candidates:]
                    cand_anom = set()
                    for num_cand in num_cand_no_anom:
                        if label not in predicted[-num_cand:]:
                            TP[num_cand] += test_abnormal_loader[line]
                            cand_anom.add(num_cand)
                        #break
                    num_cand_no_anom = num_cand_no_anom.difference(cand_anom)
                    if len(num_cand_no_anom) == 0:
                    #    print("Breaking")
                        break
                #if len(num_cand_no_anom) != 0:
                #    print("No break: " + str(num_cand_no_anom))

        # Compute precision, recall and F1-measure
        FN = {}
        TN = {}
        P = {}
        R = {}
        F1 = {}
        max_f1 = None
        best_num_cand = None
        for num_cand in set(range(1, max_num_classes)):
            FN[num_cand] = test_abnormal_length - TP[num_cand]
            TN[num_cand] = test_normal_length - FP[num_cand]
            P[num_cand] = 100 * TP[num_cand] / (TP[num_cand] + FP[num_cand])
            R[num_cand] = 100 * TP[num_cand] / (TP[num_cand] + FN[num_cand])
            F1[num_cand] = 2 * P[num_cand] * R[num_cand] / (P[num_cand] + R[num_cand])
            if max_f1 is None or F1[num_cand] > max_f1:
                max_f1 = F1[num_cand]
                best_num_cand = num_cand
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP[best_num_cand], FN[best_num_cand], P[best_num_cand], R[best_num_cand], F1[best_num_cand]))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        return print_results(TP[best_num_cand], FN[best_num_cand], TN[best_num_cand], FP[best_num_cand], best_num_cand, elapsed_time)

    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.data_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
