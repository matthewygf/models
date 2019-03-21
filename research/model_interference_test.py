import os 
import time
import subprocess
import datetime
import time
import logging

def main():
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(project_dir, execution_id)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'output.log') 
    tf_out_file = os.path.join(output_dir, 'tf_out.log')
    err_out_file = os.path.join(output_dir, 'err.log') 
    tf_log = logging.getLogger('tensorflow')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(tf_out_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    tf_log.addHandler(fh) 
    err = open(err_out_file, 'w+')
    with open(output_file, 'w+') as out:
        try:
            start_time = time.time()
            p = subprocess.Popen(
                ['python3', 
                'slim/train_image_classifier.py', 
                '--dataset_name', 
                'cifar10',
	        '--dataset_dir',
		'/datasets/cifar10',	
		'--model_name',
		'mobilenet_v2_035',	
		'--train_dir', '/experiment/mobilenet_v2_035'],
                stdout=out,
                stderr=err)
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
            print('killing')
        finally:
            out.close()
            err.close()   
            wall_time = time.time() - start_time
            print(wall_time)

if __name__ == "__main__":
    main()
        
