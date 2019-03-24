import os 
import time
import subprocess
import datetime
import time
import logging
import psutil
from multiprocessing import Process

mobile_net_v2_035_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v2_035',	
                         '--batch_size', '16']
mobile_net_v1_025_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v1_025',
                         '--batch_size', '32',
                         ]
ptb_word_lm_cmd = ['python3', 'tutorials/rnn/ptb/ptb_word_lm.py',
                   '--data_path','/models/simple-examples/data/',
                   '--model','small',
                   '--rnn_mode', 'cudnn'
                  ]
models_train = {
    'mobilenet_v2_035': mobile_net_v2_035_cmd,
    'mobilenet_v1_025': mobile_net_v1_025_cmd,
    'ptb_word_lm': ptb_word_lm_cmd
}

def process(line):
    # assuming slim learning
    # assuming have sec/step
    if 'sec/step' in line:
        return line.split('(', 1)[1].split('sec')[0]
    else:
        return 0.0

def get_average_num_step(file_path):
    num = 0.0
    mean = 0.0
    with open(file_path, 'r') as f:
        for line in f:
            mean = mean * num
            time_elapsed = process(line)
            num += 1
            mean = (mean + float(time_elapsed)) / num
    return (num, mean)

def create_process(model_name, index, percent=0.99):
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(project_dir, execution_id+model_name+str(index))
    output_file = os.path.join(output_dir, 'output.log') 
    err_out_file = os.path.join(output_dir, 'err.log') 
    train_dir = os.path.join(output_dir, 'experiment')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    err = open(err_out_file, 'w+')
    out = open(output_file, 'w+')
    cmd = models_train[model_name]
    train_dir_path = '--train_dir' if 'word' not in model_name else '--save_path' 
    cmd += [train_dir_path, train_dir, '--gpu_memory_fraction', str(percent)]
    p = subprocess.Popen(cmd, stdout=out, stderr=err)
    return (p, out, err, err_out_file)
    

def main():
    # which one we should run in parallel
    models = ['ptb_word_lm']
    processes_list = []
    err_logs = []
    out_logs = []
    err_file_paths = []
    start_times = []
    percent = (1 / len(models)) - 0.075 # some overhead of cuda stuff i think :/
    for i, m in enumerate(models):
        start_time = time.time()
        p, out, err, path = create_process(m, i, percent)
        processes_list.append(p)
        err_logs.append(err)
        out_logs.append(out)
        start_times.append(start_time)
        err_file_paths.append(path)
    should_stop = False

    try:
        while not should_stop:
            time.sleep(5)
            if len(processes_list) <= 0:
                should_stop = True

            for i,(p, err, out, start_time, path) in enumerate(zip(processes_list, err_logs, out_logs, start_times, err_file_paths)):
                poll = None
                pid = p.pid
                if poll is None:
                    print('Process %d still running' % pid)
                current_time = time.time()
                executed = current_time - start_time
                print("checking the time, process %d been running for %d " % (pid,executed))
                if executed >= 60.0 * 5:
                    p.kill()
                    err.close()
                    out.close()
                    path_i = path
                    print(path_i) 
                    num, mean = get_average_num_step(path_i)
                    processes_list.pop(i)
                    err_logs.pop(i)
                    out_logs.pop(i)
                    start_times.pop(i)
                    err_file_paths.pop(i)
                    print("%d process average num p step is %.4f and total number of step is: %d" % (pid, mean, num))
        print('finished')
    except KeyboardInterrupt:
        for p, err, out in zip(processes_list, err_logs, out_logs):
            pid = p.pid
            p.kill()
            err.close()
            out.close()
            print('%d killed ! ! !' % pid)

if __name__ == "__main__":
    main()
        
