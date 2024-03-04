import deeplog
import loganomaly
import loganomalysem
import sample
import sample_shuffle
import time

def repeat(data_dir, model, num_classes, num_candidates, iterations, train_ratio, out):
    dataset = data_dir.split('/')[-2]
    for i in range(iterations):
        print(data_dir + ': Iteration ' + str(i) + ' ... ')
        if dataset == "hdfs_logdeep":
            # Leave as is
            pass
        elif dataset == "thunderbird_cfdr":
            # If you run into a CUDA memory error when processing Thunderbird, you may have to run the following command:
            # export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
            # Use the following for sampling when sequence identifiers are used
            sample.do_sample('/home/ubuntu/logdeep/data/' + dataset, 0.2, 0.05, None) # Since only a n-th of the data is selected, need to take n * 1% of training data to train on the same number of sequences
            # Use the following for sampling when time-windows are used
            #sample.do_sample('/home/ubuntu/logdeep/data/' + dataset, 0.2, 0.05, 1800)
        else:
            # Use the following for shuffling when sequence identifiers are used
            sample_shuffle.do_sample('/home/ubuntu/logdeep/data/' + dataset, train_ratio)
            # Use the following for sampling when time-windows are used
            #sample.do_sample('/home/ubuntu/logdeep/data/' + dataset, train_ratio, 1, 1800)
            #sample.do_sample('/home/ubuntu/logdeep/data/' + dataset, train_ratio, 1, None)
        time.sleep(5)
        if model == "deeplog":
            deeplog.options["data_dir"] = data_dir
            deeplog.options["num_classes"] = num_classes
            deeplog.options["num_candidates"] = num_candidates
            if dataset == "thunderbird_cfdr":
                loganomaly.options["batch_size"] = 512
                loganomaly.options["max_epoch"] = 100
            print(deeplog.options)
            deeplog.train()
            results = deeplog.predict()
        elif model == "loganomaly":
            loganomaly.options["data_dir"] = data_dir
            loganomaly.options["num_classes"] = num_classes
            loganomaly.options["num_candidates"] = num_candidates
            if dataset == "thunderbird_cfdr":
                loganomaly.options["batch_size"] = 512
                loganomaly.options["max_epoch"] = 100
            print(loganomaly.options)
            loganomaly.train()
            results = loganomaly.predict()
        elif model == "loganomalysem":
            loganomalysem.options["data_dir"] = data_dir
            loganomalysem.options["num_classes"] = num_classes
            loganomalysem.options["num_candidates"] = num_candidates
            print(loganomalysem.options)
            loganomalysem.train()
            results = loganomalysem.predict()
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
    repeat("../data/bgl_cfdr/", "deeplog", 401, 100, 25, 0.01, out)
    #repeat("../data/bgl_cfdr/", "loganomaly", 401, 100, 25, 0.01, out)
    #repeat("../data/hdfs_xu/", "loganomalysem", 33, 10, 1, 0.01, out)
    #repeat("../data/hdfs_logdeep/", "deeplog", 28, 50, 1, None, out)
    #repeat("../data/hdfs_logdeep/", "loganomaly", 28, 50, 1, None, out)
    #repeat("../data/bgl_cfdr/", "deeplog", 401, 50, 25, 0.01, out)
    #repeat("../data/bgl_cfdr/", "loganomaly", 401, 9, 25, 0.01, out)
    #repeat("../data/thunderbird_cfdr/", "deeplog", 6425, 100, 25, 0.1, out)
    #repeat("../data/thunderbird_cfdr/", "loganomaly", 6425, 100, 25, 0.1, out)
    #repeat("../data/hadoop_loghub/", "deeplog", 349, 30, 25, 0.1, out)
    #repeat("../data/hadoop_loghub/", "loganomaly", 349, 30, 25, 0.1, out)
    #repeat("../data/adfa_verazuo/", "deeplog", 340, 100, 25, 0.01, out)
    #repeat("../data/adfa_verazuo/", "loganomaly", 340, 100, 25, 0.01, out)
    #repeat("../data/hdfs_xu/", "deeplog", 33, 30, 25, 0.01, out)
    #repeat("../data/hdfs_xu/", "loganomaly", 33, 30, 25, 0.01, out)
    #repeat("../data/bgl_cfdr/", "deeplog", 401, 100, 25, 0.01, out)
    #repeat("../data/bgl_cfdr/", "loganomaly", 401, 100, 25, 0.01, out)

print('Done ' + str(time.strftime("%Y %m %d %H:%M:%S", time.localtime())))
