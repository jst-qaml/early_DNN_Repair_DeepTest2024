import sys
import importlib
import os
import traceback
import json
import multiprocessing as mp
import numpy as np

from glob import glob
from time import sleep
from pathlib import Path

from utils import _get_dataset, _get_model, _get_repair_method, get_metrics, get_available_gpus, get_gpu_memory

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["KMP_AFFINITY"] = "noverbose"
# os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

tf = importlib.import_module('tensorflow')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# fix random seed for reproducibility
seed = 42
# tf.random.set_seed(seed)
train_seeds = [1024, 2048, 4096, 8192, 16384]

# Setup GPUs
num_gpus = 1
INITIAL_GPU = 0
GPU_PROCESS_LIMIT = 1500 # MB

# Choose datasets
divide_rate = 0.3
random_state = seed

# Choose models
# dataset_names = ['GTSRB']
# models = ['vgg16_fine_tuning']
# epoch_num = 30
# REPAIR_CLS = 27


dataset_names = ['RSNA_small']
models = ['densenet121_no_weights']
epoch_num = 30
REPAIR_CLS = 1

# Choose FL methods
FL_methods = {
    'Arachne' : [
                {'num_grad':None,'num_particles':100,'num_iterations':50,'num_input_pos_sampled':200,'velocity_phi':4.1,'min_iteration_range':10,'batch_size':32},
            ],
    }

# NOTE Important to keep tensorflow CUDA initialization inside subprocess, else this doesn't work
def train_model(model_name, rank, output_dir, kwargs, train_seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(range(num_gpus)[rank])
 
    while(get_gpu_memory()[rank] > GPU_PROCESS_LIMIT):
        print(f'sleep proc in rank {rank}')
        sleep(10)
    print(f'Training on rank {rank} is starting')

    import tensorflow as tf
    import numpy as np

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_virtual_device_configuration(gpus[rank], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
    # except RuntimeError as e:
    #     print(e)

    model = _get_model(model_name, kwargs=kwargs)

    try:
        # Use seeds to make training deterministic
        # g = tf.Graph()
        # with g.as_default():
        tf.random.set_seed(train_seed)
        tf.config.experimental.enable_op_determinism()
        np.random.seed(train_seed)
        dataset.train(model,
                    kwargs['batch_size'],
                    kwargs['epochs'],
                    kwargs['validation_split'],
                    kwargs['gpu'],
                    data_dir = kwargs['data_dir'],
                    output_dir = output_dir)
    except Exception as e:
        print(f'[EXCEPTION] during model {model_name} train step... error: \n{e}')
        print(traceback.format_exc())


def test_model(model_dir, data_dir, rank, target_data='test.h5', batch_size=64, score_dict=None):
    
    if score_dict is None:
        print(f'[WARNING] The models test scores will not be stored or shown')
    else:
        model_name = os.path.basename(os.path.dirname(model_dir))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(range(num_gpus)[rank])
 
    while(get_gpu_memory()[rank] > GPU_PROCESS_LIMIT):
        print(f'sleep proc in rank {rank}')
        sleep(30)
    print(f'Testing on rank {rank} is starting')

    import tensorflow as tf

    score = None
    try:
        score = dataset.test(model_dir, data_dir, target_data, batch_size)
    except Exception as e:
        print(f'[EXCEPTION] during model {os.path.basename(os.path.dirname(model_dir))} test step: \n{e}')
        print(traceback.format_exc())

    score_dict[model_name] = score

def target_30(rank,
           model_dir,
           data_dir,
           batch_size=64,
           dataset_type='repair',
           do_cleanup=True):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(range(num_gpus)[rank])
 
    while(get_gpu_memory()[rank] > GPU_PROCESS_LIMIT):
        print(f'sleep proc in rank {rank}')
        sleep(30)
    print(f'Target on rank {rank} is starting')

    import tensorflow as tf

    try:
        import shutil
        from dataset.test import load_model, _load_dataset, _parse_test_results, _save_positive_results, _save_negative_results, _save_label_data

        check_point = model_dir + 'logs/' + 'model_check_points/'
        target_output_path = model_dir + 'targets/'

        all_weights = Path(check_point).glob('weights*.hdf5')
        all_weights = [str(p) for p in list(all_weights)]
        # print(all_weights)
        all_weights = sorted((list(all_weights)), key=lambda i: int(os.path.basename(i)[8:10]))
        print(all_weights)
        for i, model_dir in enumerate(all_weights):

            model_weights_epoch = os.path.basename(model_dir)[8:10]

            if i != len(all_weights)-1:
                if model_weights_epoch not in ['01','05','10','15','20','25','30']:
                    continue

            # Copy model to rep folder
            save_path = target_output_path + 'ep' + model_weights_epoch + '/'
            try:
                Path(save_path).mkdir(parents=True, exist_ok=True)
            except:
                print(f'Folder already exists, skip making it')
            model_d = save_path + os.path.basename(model_dir)
            shutil.copyfile(model_dir, model_d)

            data_dir = Path(data_dir)

            # Load DNN model
            model = load_model(model_dir)
            # Load test images and labels
            datasets = _load_dataset(data_dir, r'repair.h5')
            test_images, test_labels = datasets.get_generators()

            # Predict labels from test images
            results = model.predict(test_images, verbose=0, batch_size=batch_size)

            # Parse and save predict/test results
            print("parse test")
            test_images, test_labels = datasets.get_generators()
            successes, failures = _parse_test_results(test_images,
                                                    test_labels,
                                                    results)
            print(f"save positive in model file {save_path}")
            _save_positive_results(successes, Path(''), save_path + r'positive/', dataset_type, do_cleanup)
            _save_negative_results(failures, Path(''), save_path + r'negative/', dataset_type, do_cleanup)

            _save_label_data(successes, Path('').joinpath(save_path + r'positive/labels.json'))
            _save_label_data(failures, Path('').joinpath(save_path + r'negative/labels.json'))
            

    except Exception as e:
        print(f'[EXCEPTION] during model {os.path.basename(os.path.dirname(model_dir))} target step: \n{e}')
        print(traceback.format_exc())


def evaluate(rank, method, model_dir, target_data_dir, positive_inputs_dir, output_dir = None, num_runs = 1, verbose = 1, method_config=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(range(num_gpus)[rank])
 
    while(get_gpu_memory()[rank] > GPU_PROCESS_LIMIT):
        print(f'sleep proc in rank {rank}')
        sleep(20)
    print(f'Evaluate on rank {rank} is starting')

    import tensorflow as tf

    ####
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_virtual_device_configuration(gpus[rank], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])
    # except RuntimeError as e:
    #     print(e)

    if method == 'Athena':
        print(f'Athena evaluation is not implemented. Use Arachne instead.')
        method = 'Arachne'

        
    method = _get_repair_method(method, dataset, method_config)
    if output_dir is None:
        output_dir = target_data_dir

    # Load
    target_data = dataset.load_repair_data(target_data_dir)
    positive_inputs = dataset.load_repair_data(positive_inputs_dir)

    print(f'Params: \nmethod: {method}\nmodel_dir: {model_dir}\ntarget_data_dir: {target_data_dir}\npositive_inputs_dir: {positive_inputs_dir}\noutput_dir: {output_dir}\nmethod_config {method_config}')

    # Evaluate
    method.evaluate(dataset,
                    method,
                    model_dir,
                    target_data,
                    target_data_dir,
                    positive_inputs,
                    positive_inputs_dir,
                    output_dir,
                    num_runs,
                    verbose)


def metrics(rank, model_dir, test_data_path, target_data_dir, positive_data_dir, repaired_model):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(range(num_gpus)[rank])
 
    while(get_gpu_memory()[rank] > GPU_PROCESS_LIMIT):
        print(f'sleep proc in rank {rank}')
        sleep(30)
    print(f'Evaluate on rank {rank} is starting')

    import tensorflow as tf
    gpu = get_available_gpus()[0]

    ####
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_virtual_device_configuration(gpus[rank], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])
    # except RuntimeError as e:
    #     print(e)

    model = dataset.load_model(model_dir)

    metrics_dict = get_metrics(model, test_data_path, target_data_dir, positive_data_dir, repaired_model)

    # Save dict in json
    if not os.path.isdir(model_dir):
        dump_path = os.path.dirname(model_dir) + '/'
    else:
        dump_path = model_dir
    with open(dump_path + 'test_class_metrics.txt', 'w') as convert_file:
     convert_file.write(json.dumps(metrics_dict))

    # Pretty print results
    model = os.path.basename(model_dir)
    acc = metrics_dict['accuracy']
    f1 = metrics_dict['f1']
    recall = metrics_dict['recall']
    auc = metrics_dict['auc_under_roc']

    print(f'Model: {model_dir}\nAccuracy: {acc}\nF1 Score: {f1}\nRecall: {recall}\nAUC of ROC: {auc}')
    if repaired_model:
        rr = metrics_dict['rr']
        br = metrics_dict['br']
        print(f'RR: {rr}\nBR: {br}')
    
    return

if __name__ == '__main__':
    base_input_path = Path(os.getcwd()+'/inputs/')
    base_output_path = Path(os.getcwd()+'/outputs/')
    print(f'Dataset input path {str(base_input_path)}')

    # For each dataset create its own results folder
    for dataset_name in dataset_names:

        print(f'------------------- Dataset initializations {dataset_name} -------------------')
        if dataset_name == 'GTSRB':
            dataset_input_path = base_input_path.joinpath('gtsrb/')
        elif dataset_name == 'RSNA_small':
            dataset_input_path = base_input_path.joinpath('RSNA_pneuomonia_detection/')
        else:
            dataset_input_path = base_input_path
        # dataset_input_path.mkdir(parents=True, exist_ok=True)
        dataset_output_path = base_output_path / dataset_name.lower()
        dataset_output_path.mkdir(parents=True, exist_ok=True)

        # Prepare dataset
        dataset_kwargs = {'input_dir': dataset_input_path,
                'output_dir': dataset_output_path,
                'divide_rate': divide_rate,
                'random_state': random_state}
        
        dataset = _get_dataset(dataset_name, kwargs=dataset_kwargs)

        # If dataset was not preprocessed, do it
        if not dataset_output_path.joinpath(r'train.h5').exists() \
            or not dataset_output_path.joinpath(r'test.h5').exists() \
            or not dataset_output_path.joinpath(r'repair.h5').exists():
            
            print(f'Pre-processing was not done for this dataset ({dataset_name}). Commence pre-processing...')

            dataset.prepare(dataset_input_path, dataset_output_path, divide_rate, random_state)

        print(f'------------------- Model initializations and trainings -------------------')
        
        for i, train_seed in enumerate(train_seeds):
            # Setup paths for model trainings
            model_output_paths = []
            for model_name in models:
                new_model_name  = model_name + f'_run{i}'
                model_output_paths.append(dataset_output_path / new_model_name / f'epoch{epoch_num}')

            print(f'Model names of the template: {model_output_paths[0]}')

            for path in model_output_paths:
                path.mkdir(parents=True, exist_ok=True)

            # Spawn number of processes according to number of models
            model_kwargs = [{'batch_size': 128,
                        'epochs': epoch_num,
                        'validation_split': 0.2,
                        'gpu': True,
                        'data_dir': str(dataset_output_path) + '/',}]
            
        # Setup parameters for multiprocess
        mp_params = []
        rank = INITIAL_GPU

        # For all models, for all epoch trainings
        epoch_idx = 0
        for i, model_epoch_path in enumerate(model_output_paths):
            mp_params.append((str(model_epoch_path.parents[0].name[:-5]), rank, str(model_epoch_path), model_kwargs[epoch_idx], train_seeds[i]))
            epoch_idx = 0 if epoch_idx == len(model_kwargs) - 1 else epoch_idx + 1
            rank = INITIAL_GPU if rank == num_gpus - 1 else rank + 1 

        print(f'Training params len: {len(mp_params)} and : {mp_params}')

        print(f'Start trainings...')
        count = 0
        running_tasks = [mp.Process(target=train_model, args=param) for param in mp_params]
        for i, running_task in enumerate(running_tasks):
            running_task.daemon = False
            running_task.start()
            sleep(30+count*3)
            count += 1
        for running_task in running_tasks:
            running_task.join()



        model_output_paths = []
        for i in range(len(train_seeds)):
            for model_name in models:
                new_model_name  = model_name + f'_run{i}'
                model_output_paths.append(dataset_output_path / new_model_name / f'epoch{epoch_num}')

        model_kwargs = []
        for i in range(len(train_seeds)):
            model_kwargs.extend([{'batch_size': 32,
                        'epochs': epoch_num,
                        'validation_split': 0.2,
                        'gpu': True,
                        'data_dir': str(dataset_output_path) + '/',}])
            


        print(f'------------------- Target 30 -------------------')

        # Setup parameters for multiprocess
        mp_params = []
        rank = INITIAL_GPU
        for i, model_epoch_path in enumerate(model_output_paths):
            mp_params.append((rank, str(model_epoch_path) + '/', str(dataset_output_path), 2))
            rank = INITIAL_GPU if rank == num_gpus - 1 else rank + 1 

        print(f'mp params at target: {mp_params}')

        print(f'Target dataset...')
        running_tasks = [mp.Process(target=target_30, args=param) for param in mp_params] 
        i=1
        for running_task in running_tasks:
            running_task.daemon = False
            running_task.start()
            sleep(10 + i*3)
            i += 1
        for running_task in running_tasks:
            running_task.join()

        rep_paths = []
        rep_model_paths = []
        for model_dir in model_output_paths:
            target_output_path = model_dir / 'targets'
            paths = [str(x)+'/' for x in list(target_output_path.glob('*/'))]
            rep_model_p = [str(list(Path(p).glob('*.hdf5'))[0]) for p in paths]
            rep_paths.extend(paths)
            rep_model_paths.extend(rep_model_p)

        print(len(rep_paths) == len(rep_model_paths))

        print(f'------------------- Evaluate -------------------')
        

        # Setup output directories
        optimize_output_paths = [str(path) + f'/negative/{REPAIR_CLS}/' for path in rep_paths]

        # Setup parameters for multiprocess - do combinations of models and FL methods
        mp_params = [] 
        rank = INITIAL_GPU
        for i, model_output_path in enumerate(rep_paths):
            for method in FL_methods.keys():
                print(str(str(model_output_path).split('/')[-2]))
                mp_params.extend([(rank, method, rep_model_paths[i], str(model_output_path)+ f'/negative/{REPAIR_CLS}/', str(model_output_path)+ '/positive/', \
                                    optimize_output_paths[i], 1, 1, FL_methods[method][config_id]) for config_id in range(len(FL_methods[method]))])
                rank = INITIAL_GPU if rank == num_gpus - 1 else rank + 1

        print(f'Multiproc params for optimization are of length: {len(mp_params)}')

        print(f'Evaluate...')
        i=1
        running_tasks = [mp.Process(target=evaluate, args=param) for param in mp_params]
        for running_task in running_tasks:
            running_task.daemon = False
            running_task.start()
            sleep(60 + i*7)
            i += 1
        for running_task in running_tasks:
            running_task.join()


        print(f'------------------- Metrics -------------------')

        # Setup repaired model's paths
        repaired_model_paths = [] 
        target_data_dirs = []
        positive_data_dirs = []
        repaired_models = []

        for path in rep_model_paths:
            repaired_model_paths.append(str(path))
            target_data_dirs.append(None)
            positive_data_dirs.append(None)
            repaired_models.append(False)

        for optimize_output_path in optimize_output_paths:
            repaired_model_path_list = glob(optimize_output_path + 'repaired_model_[0-9]/repair/')
            repaired_model_paths.extend(repaired_model_path_list)
            repaired_model_path_list2 = glob(optimize_output_path + 'repair')
            repaired_model_paths.extend(repaired_model_path_list2)

            # Add paths for RR and BR
            target_data_dirs.extend([optimize_output_path for i in range(len(repaired_model_path_list))])
            target_data_dirs.extend([optimize_output_path for i in range(len(repaired_model_path_list2))])
            positive_data_dirs.extend([optimize_output_path[:-12]+'positive/' for i in range(len(repaired_model_path_list))])
            positive_data_dirs.extend([optimize_output_path[:-12]+'positive/' for i in range(len(repaired_model_path_list2))])

            repaired_models.extend([True for i in range(len(repaired_model_path_list))])
            repaired_models.extend([True for i in range(len(repaired_model_path_list2))])

        print(f'Optimize_out paths {len(optimize_output_paths)}: {optimize_output_paths}')
        print(f'Repaired model paths {len(repaired_model_paths)}: {repaired_model_paths}')
        print(f'Target data dirs {len(target_data_dirs)}: {target_data_dirs}')
        print(f'Positive data dirs {len(positive_data_dirs)}: {positive_data_dirs}')
        print(f'Required RR/BR {len(repaired_models)}: {repaired_models}')

        test_data_path = str(dataset_output_path.joinpath(r'test.h5'))
        print(f'Test data path: {test_data_path}')
        # Setup parameters for multiprocess - do combinations of models and FL methods
        mp_params = [] 
        rank = INITIAL_GPU
        for i, model_path in enumerate(repaired_model_paths):
            mp_params.append((rank, model_path, test_data_path, target_data_dirs[i], positive_data_dirs[i], repaired_models[i]))
            rank = INITIAL_GPU if rank == num_gpus - 1 else rank + 1

        print(f'Multiproc params for optimization are of length: {len(mp_params)}')

        print(f'Calculate metrics...')
        i = 1
        running_tasks = [mp.Process(target=metrics, args=param) for param in mp_params]
        for running_task in running_tasks:
            running_task.daemon = False
            running_task.start()
            sleep(10 + i*3)
            i += 1
        for running_task in running_tasks:
            running_task.join()