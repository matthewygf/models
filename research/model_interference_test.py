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
                         '--batch_size', '16',
                         ]
lenet_cmd = ['python3', 'slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'lenet',	
                         '--train_dir', '/experiment/lenet/',
                         '--batch_size', '16']
models_train = {
    'mobilenet_v2_035': mobile_net_v2_035_cmd,
    'mobilenet_v1_025': mobile_net_v1_025_cmd,
    'lenet': lenet_cmd
}

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

def main():
    # which one we should run in parallel
    models = ['mobilenet_v1_025', 'mobilenet_v1_025']
    processes_list = []
    percent = (1 / len(models)) - 0.015 # some overhead of cuda stuff i think :/
    for i, m in enumerate(models):
        p = Process(target=create_process, args=(m, i, percent))
        processes_list.append(p)
    try:
        for p in processes_list:
            p.start()
            time.sleep(5)
    except KeyboardInterrupt:
        for p in processes_list:
            p.terminate()
    finally:
        print("finishhhhh launchingggg!")

if __name__ == "__main__":
    main()
        
