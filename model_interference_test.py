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
mobile_net_v1_025_b48_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v1_025',
                         '--batch_size', '48',
                         ]
nasnet_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'nasnet_cifar',	
                         '--batch_size', '8']


resnet_50_b32_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'resnet_v2_50',	
                         '--batch_size', '32']
mobile_net_v1_025_b32_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v1_025',
                         '--batch_size', '32',
                         ]
resnet_50_v1_b32_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'resnet_v1_50',	
                         '--batch_size', '32']
resnet_50_v1_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'resnet_v1_50',	
                         '--batch_size', '8']
vgg_19_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'vgg_19',	
                         '--batch_size', '8']
vgg_16_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'vgg_16',	
                         '--batch_size', '8']
alexnet_v2_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'alexnet_v2',	
                         '--batch_size', '8']
inception_v1_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'inception_v1',	
                         '--batch_size', '8']
inception_v2_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'inception_v2',	
                         '--batch_size', '8']
inception_v3_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'inception_v3',	
                         '--batch_size', '8']
inception_v4_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'inception_v4',	
                         '--batch_size', '8']
resnet_101_v1_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'resnet_v1_101',	
                         '--batch_size', '8']
resnet_152_v1_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'resnet_v1_152',	
                         '--batch_size', '8']


resnet_50_b8_cmd = ['python3', 'research/slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'resnet_v2_50',	
                         '--batch_size', '8']
resnet_v1_50_16_cmd = ['python3', 'research/slim/train_image_classifier.py',
                    '--dataset_name', 'imagenet',
                    '--dataset_dir', '/datasets/ILSVRC2012',
                    '--model_name', 'resnet_v1_50',
                    '--batch_size', '16']
ptb_word_lm_cmd = ['python3', 'tutorials/rnn/ptb/ptb_word_lm.py',
                   '--data_path','/models/simple-examples/data/',
                   '--model','small',
                   '--rnn_mode', 'cudnn'
                  ]
models_train = {
    'mobilenet_v2_035': mobile_net_v2_035_cmd,
    'mobilenet_v1_025_batch_40': mobile_net_v1_025_cmd,
    'mobilenet_v1_025_batch_48': mobile_net_v1_025_b48_cmd,
    'mobilenet_v1_025_batch_32': mobile_net_v1_025_b32_cmd,
    'ptb_word_lm': ptb_word_lm_cmd,
    'nasnet_batch_8': nasnet_b8_cmd,
    'resnet_50_batch8_cmd': resnet_50_b8_cmd,
    'resnet_v1_50_batch_8': resnet_50_v1_b8_cmd,
    'vgg19_batch_8': vgg_19_b8_cmd,
    'vgg16_batch_8': vgg_16_b8_cmd,
    'alexnet_v2_batch_8': alexnet_v2_b8_cmd,
    'inceptionv1_batch_8': inception_v1_b8_cmd,
    'inceptionv2_batch_8': inception_v2_b8_cmd,
    'inceptionv3_batch_8': inception_v3_b8_cmd,
    'inceptionv4_batch_8': inception_v4_b8_cmd,
    'resnet_101_v1_batch_8': resnet_101_v1_b8_cmd,
    'resnet_151_v1_batch_8': resnet_152_v1_b8_cmd
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
            if 'sec/step' in line:
                mean = mean * num
                time_elapsed = process(line)
                num += 1
                mean = (mean + float(time_elapsed)) / num
    return (num, mean)

def create_process(model_name, index, experiment_path, percent=0.0):
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
    cmd += [train_dir_path, train_dir]
    if percent > 0.0:
        gpu_mem_opts = ['--gpu_memory_fraction', str(percent)] 
        cmd += gpu_mem_opts

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

    for experiment_run in range(1, 2):
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
            smi_file_path = os.path.join(experiment_path, 'smi.log') 
            smi_file = open(smi_file_path, 'a+')
            nvidia_smi_cmd = ['watch', '-n', '0.2', 'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw', '--format=csv,noheader', '|', 'tee', '-a' , experiment_path+'/smi_watch.log']
            smi_p = subprocess.Popen(nvidia_smi_cmd, stdout=smi_file, stderr=smi_file)
            sys_tracker.start()
            while not should_stop:
                time.sleep(5)
                if len(processes_list) <= 0:
                    should_stop = True

                for i,(p, err, out, start_time, path, tracker) in enumerate(zip(processes_list, err_logs, out_logs, start_times, err_file_paths, trackers)):
                    poll = None
                    pid = p.pid
                    poll = p.poll()
                    if poll is None:
                        print('Process %d still running' % pid)
                    current_time = time.time()
                    executed = current_time - start_time
                    print("checking the time, process %d been running for %d " % (pid,executed))
                    if executed >= 60.0 * 8:
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

                smi_poll = None
                smi_poll = smi_p.poll()
                if smi_poll is None:
                    print('NVIDIA_SMI Process %d still running' % smi_p.pid)

            print('total experiments: %d, experiment_run %d , finished %d' % (total_length-1, experiment_run, experiment_index))

        except KeyboardInterrupt:
            smi_p.kill()
            smi_file.close()
            for p, err, out in zip(processes_list, err_logs, out_logs):
                pid = p.pid
                p.kill()
                err.close()
                out.close()
                print('%d killed ! ! !' % pid)
        finally:
            if smi_p.poll is None:
                smi_p.kill()
                smi_file.close()
            
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
    sets = [
            ['mobilenet_v1_025_batch_32'],
            ['mobilenet_v1_025_batch_32', 'mobilenet_v1_025_batch_32'],
            ['mobilenet_v1_025_batch_32', 'mobilenet_v1_025_batch_32', 'mobilenet_v1_025_batch_32', 'mobilenet_v1_025_batch_32'],
            ['resnet_v1_50_batch_8'], 
            ['resnet_v1_50_batch_8', 'resnet_v1_50_batch_8'],
            ['resnet_v1_50_batch_8', 'ptb_word_lm'],
            ['ptb_word_lm'],
            ['ptb_word_lm', 'ptb_word_lm'],
            ['ptb_word_lm', 'mobilenet_v1_025_batch_32'],
            ['inceptionv1_batch_8'],
            ['inceptionv1_batch_8','inceptionv1_batch_8'],
            ['resnet_v1_50_batch_8', 'inceptionv1_batch_8'], 
            ['inceptionv1_batch_8', 'ptb_word_lm']
            # ['resnet_101_v1_batch_8'],
            # ['resnet_151_v1_batch_8'],
            # ['vgg19_batch_8'], 
            # ['vgg16_batch_8'],
            # ['inceptionv1_batch_32'],
            # ['inceptionv1_batch_32', 'inceptionv1_batch_32'],
            # [ 'ptb_word_lm', 'inceptionv1_batch_32'],
            # ['inceptionv2_batch_32'],
            # ['inceptionv3_batch_32'],
            # ['inceptionv4_batch_32']
            # ['alexnet_v2_batch_32'],
           ]
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    experiment_path = os.path.join(project_dir, 'experiment')

    for experiment_index, ex in enumerate(sets):
        current_experiment_path = os.path.join(experiment_path, str(experiment_index))
        experiment_file = os.path.join(experiment_path, 'experiment.log')

        run(experiment_file, current_experiment_path, ex, len(sets), experiment_index)
if __name__ == "__main__":
    main()
        
