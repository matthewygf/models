import os 
import time
import subprocess
import datetime
import time
import logging
import psutil
import process_tracker as p_track
import system_tracker as sys_track
import numpy as np

mobile_net_v2_035_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v2_035',	
                         '--batch_size', '16']
mobile_net_v1_025_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v1_025',
                         '--batch_size', '40',
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

def create_process(model_name, index, experiment_path, percent=0.99):
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output_dir = os.path.join(experiment_path, execution_id+model_name+str(index))
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
    return (p, out, err, err_out_file, output_dir)

def run(
    average_log, experiment_path, 
    experiment_set, total_length, 
    experiment_index):
    
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    mean_num_models = np.zeros(len(experiment_set), dtype=float)
    mean_time_p_steps = np.zeros(len(experiment_set), dtype=float)
    accumulated_models = np.zeros(len(experiment_set), dtype=float)

    for experiment_run in range(1, 6):
        if os.path.exists(average_log):
            average_file = open(average_log, mode='a+')
        else:
            average_file = open(average_log, mode='w+')
        processes_list = []
        err_logs = []
        out_logs = []
        err_file_paths = []
        start_times = []
        trackers = []
        ids = {}
        percent = (1 / len(experiment_set)) - 0.075 # some overhead of cuda stuff i think :/
        for i, m in enumerate(experiment_set):
            start_time = time.time()
            p, out, err, path, out_dir = create_process(m, i, experiment_path, percent)
            tracker = p_track.ProcessInfoTracker(out_dir, p.pid)
            tracker.start()
            processes_list.append(p)
            err_logs.append(err)
            out_logs.append(out)
            start_times.append(start_time)
            err_file_paths.append(path)
            trackers.append(tracker)
            ids[p.pid] = i
        should_stop = False
        sys_tracker = sys_track.SystemInfoTracker(experiment_path)
        
        try:
            sys_tracker.start()
            while not should_stop:
                time.sleep(5)
                if len(processes_list) <= 0:
                    should_stop = True

                for i,(p, err, out, start_time, path, tracker) in enumerate(zip(processes_list, err_logs, out_logs, start_times, err_file_paths, trackers)):
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
                        model_index = ids[pid]
                        mean_num_models[model_index] = ((accumulated_models[model_index] * mean_num_models[model_index]) + num) / (accumulated_models[model_index] + 1.0)
                        mean_time_p_steps[model_index] = ((accumulated_models[model_index] * mean_time_p_steps[model_index]) + mean) / (accumulated_models[model_index] + 1.0)
                        accumulated_models[model_index] += 1.0
                        processes_list.pop(i)
                        err_logs.pop(i)
                        out_logs.pop(i)
                        start_times.pop(i)
                        err_file_paths.pop(i)
                        tracker.stop()
                        trackers.pop(i)
                        line = ("experiment set %d, experiment_run %d: %d process average num p step is %.4f and total number of step is: %d \n" % 
                                    (experiment_index, experiment_run, pid, mean, num))
                        average_file.write(line)
                        
            print('total experiments: %d, experiment_run %d , finished %d' % (total_length-1, experiment_run, experiment_index))
        except KeyboardInterrupt:
            for p, err, out in zip(processes_list, err_logs, out_logs):
                pid = p.pid
                p.kill()
                err.close()
                out.close()
                print('%d killed ! ! !' % pid)
        average_file.close()
        sys_tracker.stop()

    # Experiment average size.
    average_file = open(average_log, mode='a+')
    for i in range(len(experiment_set)):
        average_file.write("TOTAL: In experiment %d average mean sec/step and average number for model %d are %.4f , %d \n" % 
                        (experiment_index, i, mean_time_p_steps[i], mean_num_models[i]))
    average_file.close()
    
def main():
    # which one we should run in parallel
    sets = [['ptb_word_lm']]
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    experiment_path = os.path.join(project_dir, 'experiment')
    #nv_out_log = os.path.join(project_dir, 'nv_out.log')
    #nv_err_log = os.path.join(project_dir, 'nv_out.log')
    #nv_err = open(nv_err_log, 'w+')
    #nv_out = open(nv_out_log, 'w+')
    #nvprof_p = subprocess.Popen(['nvprof', '--profile-all-process', '--profile-child-processes', '--log-file', '/models/nvprof.log'], stdout=nv_out, stderr=nv_err)

    for experiment_index, ex in enumerate(sets):
        current_experiment_path = os.path.join(experiment_path, str(experiment_index))
        experiment_file = os.path.join(experiment_path, 'experiment.log')

        run(experiment_file, current_experiment_path, ex, len(sets), experiment_index)
    #nvprof_p.terminate()
    #nv_err.close()
    #nv_out.close()        

if __name__ == "__main__":
    main()
        
