import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels

def load_features(data_path, only_normal=True, min_len=0):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    if only_normal:
        logs = []
        for seq in data:
            if len(seq['EventId']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = max(seq['Label'].tolist())
            else:
                label = seq['Label']
            if label == 0:
                try:
                    logs.append((seq['EventId'], label, seq['Seq'].tolist()))
                except:
                    logs.append((seq['EventId'], label, seq['Seq']))
    else:
        logs = []
        no_abnormal = 0
        for seq in data:
            if len(seq['EventId']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = seq['Label'].tolist()
                if max(label) > 0:
                    no_abnormal += 1
            else:
                label = seq['Label']
                if label > 0:
                    no_abnormal += 1
            try:
                logs.append((seq['EventId'], label, seq['Seq'].tolist()))
            except:
                logs.append((seq['EventId'], label, seq['Seq']))
        print("Number of abnormal sessions:", no_abnormal)
    return logs

def sliding_window(data_dir, datatype, window_size, num_keys, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    #event2semantic_vec = read_json(data_dir + 'event2semantic_vec.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    #result_logs['Semantics'] = []
    labels = []
    if datatype == 'train':
        data_dir += data_dir.split('/')[-2].split('_')[0] + '_train'
    if datatype == 'val':
        data_dir += data_dir.split('/')[-2].split('_')[0] + '_test_normal'

    with open(data_dir, 'r') as f:
        for line in f.readlines():
            #if datatype == 'val' and num_sessions > 10:
            #    break
            num_sessions += 1
            if ',' in line:
                # Remove sequence identifier if available
                line = line.split(',')[1]
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            line = tuple([-1] * window_size) + line

            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * num_keys
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    #print(key)
                    try:
                        Quantitative_pattern[key] = log_counter[key]
                    except IndexError as e:
                        print(e)
                        print('Found key ' + str(key) + ' but num_keys is only ' + str(num_keys))
                #Semantic_pattern = []
                #for event in Sequential_pattern:
                #    if event <= 0:
                #        Semantic_pattern.append([-1] * 300)
                #    else:
                #        Semantic_pattern.append(event2semantic_vec[str(event - 1)])
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                  np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                #result_logs['Semantics'].append(Semantic_pattern)
                labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels


def session_window(data_dir, datatype, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + 'event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += 'robust_log_train.csv'
    elif datatype == 'val':
        data_dir += 'robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += 'robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        ori_seq = [
            int(eventid) for eventid in train_df["Sequence"][i].split(' ')
        ]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 29
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, labels
