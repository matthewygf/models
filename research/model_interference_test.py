import os 
import time
import subprocess
import datetime
import time
import logging
from multiprocessing import Process

mobile_net_v2_035_cmd = ['python3', 'slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v2_035',	
                         '--batch_size', '16']
mobile_net_v1_025_cmd = ['python3', 'slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v1_025',
                         '--batch_size', '32',
                         ]
models_train = {
    'mobilenet_v2_035': mobile_net_v2_035_cmd,
    'mobilenet_v1_025': mobile_net_v1_025_cmd,
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
    output_dir = os.path.join(project_dir, execution_id+model_name)
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
    cmd += ['--train_dir', train_dir, '--gpu_memory_fraction', str(percent)]
    try:
        start_time = time.time()
        p = subprocess.Popen(cmd, stdout=out, stderr=err)
        poll = None
        pid = p.pid
        while poll is None:
            time.sleep(5)
            print('Process %d still running' % pid)
            poll = p.poll()
        print('Process %d is finished' % pid)
        print('finished')
    except KeyboardInterrupt:
        p.kill()
        print('killed ! ! !')
    finally:
        out.close()
        err.close()   
        wall_time = time.time() - start_time
        print("%s process %d finished %s" % (model_name, index, str(wall_time)))
        num, mean = get_average_num_step(err_out_file)
        print("average num p step is %.4f" % mean)

def main():
    # which one we should run in parallel
    models = ['mobilenet_v1_025']
    processes_list = []
    start_time = time.time()
    percent = (1 / len(models)) - 0.075 # some overhead of cuda stuff i think :/
    for i, m in enumerate(models):
        p = Process(target=create_process, args=(m, i, percent))
        processes_list.append(p)
    should_stop = False
    try:
        for p in processes_list:
            p.start()
            time.sleep(5)
    except KeyboardInterrupt:
        for p in processes_list:
            p.terminate()
    finally:
        print("finishhhhh launchingggg!")
        current_time = time.time()
        while not should_stop:
            time.sleep(5)
            current_time = time.time()
            executed = current_time - start_time
            print("checking the time been running for %d " % executed)
            if executed >= 60.0 * 5:
                should_stop = True
        
        for p in processes_list:
            p.terminate()
        print("done one experiement")

if __name__ == "__main__":
    main()
        
