import deeplog
import loganomaly
import sample
import sample_shuffle
import time

def repeat(data_dir, model, num_classes, num_candidates, iterations, train_ratio, out):
    dataset = data_dir.split('/')[-2]
    for i in range(iterations):
        print(data_dir + ': Iteration ' + str(i) + ' ... ')
        if dataset == "thunderbird_cfdr":
            sample.do_sample('/home/ubuntu/logdeep/data/' + dataset, 0.2, 0.05) # Since only a n-th of the data is selected, need to take n * 1% of training data to train on the same number of sequences
        else:
            sample_shuffle.do_sample('/home/ubuntu/logdeep/data/' + dataset, train_ratio)
        #time.sleep(5)
        if model == "deeplog":
            deeplog.options["data_dir"] = data_dir
            deeplog.options["num_classes"] = num_classes
            #deeplog.options["num_candidates"] = num_candidates
            print(deeplog.options)
            deeplog.train()
            results = deeplog.predict()
        elif model == "loganomaly":
            loganomaly.options["data_dir"] = data_dir
            loganomaly.options["num_classes"] = num_classes
            loganomaly.options["num_candidates"] = num_candidates
            print(loganomaly.options)
            loganomaly.train()
            results = loganomaly.predict()
        results['name'] = model
        out_str = ""
        for csv_column in csv_columns:
            out_str += str(results[csv_column]) + ','
        out.write(dataset + ',' + str(i) + ',' + out_str[:-1] + '\n')
        out.flush()
        time.sleep(5)

csv_columns = ['tp', 'fp', 'tn', 'fn', 'tpr', 'fpr', 'tnr', 'p', 'f1', 'acc', 'threshold', 'name', 'time']
with open('runner_result3.csv', 'w+') as out:
    out.write('data_dir,iteration,' + ','.join(csv_columns) + '\n')
    #repeat("../data/thunderbird_cfdr/", "deeplog", 6425, 100, 25, 0.1, out)
    repeat("../data/thunderbird_cfdr/", "loganomaly", 6425, 100, 25, 0.1, out)
    #repeat("../data/hadoop_loghub/", "deeplog", 349, 30, 25, 0.1, out)
    #repeat("../data/hadoop_loghub/", "loganomaly", 349, 30, 25, 0.1, out)
    #repeat("../data/adfa_verazuo/", "deeplog", 340, 30, 25, 0.01, out)
    #repeat("../data/adfa_verazuo/", "loganomaly", 340, 30, 25, 0.01, out)
    #repeat("../data/hdfs_xu/", "deeplog", 33, 9, 25, 0.01, out)
    #repeat("../data/hdfs_xu/", "loganomaly", 33, 9, 25, 0.01, out)
    #repeat("../data/bgl_cfdr/", "deeplog", 401, 50, 25, 0.01, out)
    #repeat("../data/bgl_cfdr/", "loganomaly", 401, 9, 25, 0.01, out)

print('Done ' + str(time.strftime("%Y %m %d %H:%M:%S", time.localtime())))
