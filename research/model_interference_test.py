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
                         '--train_dir', '/experiment/mobilenet_v2_035/',
                         '--batch_size', '16']
mobile_net_v1_025_cmd = ['python3', 'slim/train_image_classifier.py', 
                         '--dataset_name', 'cifar10',
                         '--dataset_dir', '/datasets/cifar10',
                         '--model_name', 'mobilenet_v1_025',	
                         '--train_dir', '/experiment/mobilenet_v1_025/',
                         '--batch_size', '16']
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

def create_process(model_name, index):
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(project_dir, execution_id)
    output_dir = os.path.join(output_dir, model_name)
    output_file = os.path.join(output_dir, 'output.log') 
    err_out_file = os.path.join(output_dir, 'err.log') 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    err = open(err_out_file, 'w+')
    out = open(output_file, 'w+')
    try:
        start_time = time.time()
        p = subprocess.Popen(models_train[model_name], stdout=out, stderr=err)
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
    models = ['lenet', 'lenet']
    processes_list = []
    for i, m in enumerate(models):
        p = Process(target=create_process, args=(m, i))
        processes_list.append(p)
    try:
        for p in processes_list:
            p.start()
    except KeyboardInterrupt:
        for p in processes_list:
            p.terminate()
    finally:
        print("finishhhhh launchingggg!")

if __name__ == "__main__":
    main()
        
