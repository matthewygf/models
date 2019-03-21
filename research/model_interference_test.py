import os 
import time
import subprocess
import datetime
import time

def main():
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(project_dir, execution_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'output.log')
    with open(output_file, 'w+') as out:
        try:
            p = subprocess.Popen(
                ['python3', 
                'slim/train_image_classifier.py', 
                '--dataset_name', 
                'cifar10'],
                stdout=out)
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

if __name__ == "__main__":
    main()
        