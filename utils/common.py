# System
import importlib
import os
import sys
import shutil
import copy
import yaml
from colorama import Fore
import csv

class DotDict(dict):
    """
    Dot notation access to dictionary attributes, recursively.
    """
    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    __setattr__ = dict.__setitem__

    def __delattr__(self, attr):
        del self[attr]

    def __missing__(self, key):
        self[key] = DotDict()
        return self[key]


  
def model_class_from_str(model_name, model_type=None):
    """
    Retrieve the model class based on the provided model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        class: The model class corresponding to the provided model name.

    Example:
        >>> model = model_class_from_str('encoder1')
        >>> model_instance = model()
        >>> model_instance.train()
    """
    if model_type is None:
        module_name = importlib.import_module(f'models.{model_name}')
    else:
        module_name = importlib.import_module(f'models.{model_type}.{model_name}')
    model_class = getattr(module_name, model_name)
    assert callable(model_class)
    return model_class


def class_from_str(module, class_type):
    """
    Retrieve the model class based on the provided model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        class: The model class corresponding to the provided model name.

    Example:
        >>> model = model_class_from_str('encoder1')
        >>> model_instance = model()
        >>> model_instance.train()
    """

    module_name = importlib.import_module(module)
    
    model_class = getattr(module_name, class_type)
    assert callable(model_class)
    return model_class

def setup_experiment(args: dict, file: str = "train.yaml") -> dict:
    # Check if experiment_name is provided, otherwise assume it as "baseline"
    if args["experiment_name"] is None:
        args["experiment_name"] = "baseline"
        print(f"{Fore.YELLOW}Missing input 'experiment_name'. Assumed to be 'baseline'{Fore.RESET}")
    
    # Check if overwrite flag is provided
    overwrite = args['overwrite']
    experiment_name = args['experiment_name']

    # Load train config
    PHD_ROOT = os.getenv("PHD_ROOT")
    PHD_RESULTS = os.getenv("PHD_RESULTS")
    sys.path.append(PHD_ROOT)

    experiment_base = None
    for folder in os.listdir(f"{PHD_ROOT}/multi_task_RL/experiments/"):
        print(folder)
        for experiment in os.listdir(f"{PHD_ROOT}/multi_task_RL/experiments/{folder}"):
            if experiment == experiment_name: experiment_base = folder
        
        if experiment_base is not None: break 

    cfg_path = f"{PHD_ROOT}/multi_task_RL/experiments/{experiment_base}/{experiment_name}/{file}"
    with open(cfg_path) as f:
        cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))

    # Check if identifier is provided, otherwise set it as "1"
    if args["identifier"] == "":
        args["identifier"] = "1"
    elif args["identifier"] == "auto":
        # Find the latest experiment folder and increment the identifier
        if not os.path.exists(f'{PHD_RESULTS}/models/{cfg.project}/{experiment_name}'):
            os.makedirs(f'{PHD_RESULTS}/models/{cfg.project}/{experiment_name}')
        files = os.listdir(f'{PHD_RESULTS}/models/{cfg.project}/{experiment_name}')
        folder_experiments = [int(s) for s in files if s.isdigit()]

        if len(folder_experiments) == 0:
            args["identifier"] = 1
        else:
            folder_experiments.sort()
            args["identifier"] = folder_experiments[-1] + 1 
            
    experiment_path = f'{PHD_RESULTS}/models/{cfg.project}/{experiment_name}/{args["identifier"]}'    
    log_path = f'{PHD_RESULTS}/logs/{cfg.project}/{experiment_name}/{args["identifier"]}'    
    
    if not args['debug']:
        if os.path.exists(experiment_path):
            if overwrite:
                shutil.rmtree(experiment_path)
                shutil.rmtree(log_path)
                print(f'Removing original {experiment_path}')
                print(f'Removing original {log_path}')
            else:
                print(f'{experiment_path} already exits. ')
                raise Exception('Experiment name already exists. If you want to overwrite, use flag -ow')

        # Create folder to store the results
        os.makedirs(experiment_path)
        os.makedirs(f"{experiment_path}/best_model")
        print(f"Path create: {experiment_path}") 
        shutil.copy(cfg_path, f"{experiment_path}/train.yaml")

        # Create folder to store the results
        os.makedirs(log_path)
        print(f"Path create: {log_path}") 
        shutil.copy(cfg_path, f"{log_path}/train.yaml")
    
    return experiment_name, experiment_path, log_path, cfg

def setup_test(args: dict) -> dict:
    # load train config.
    PHD_ROOT = os.getenv("PHD_ROOT")
    PHD_RESULTS = os.getenv("PHD_RESULTS")
    sys.path.append(PHD_ROOT)
    experiment_name = args['experiment_name']

    experiment_base = None
    for folder in os.listdir(f"{PHD_ROOT}/multi_task_RL/experiments/"):
        print(folder)
        for experiment in os.listdir(f"{PHD_ROOT}/multi_task_RL/experiments/{folder}"):
            if experiment == experiment_name: experiment_base = folder
        if experiment_base is not None: break 


    cfg_path = f"{PHD_ROOT}/multi_task_RL/experiments/{experiment_base}/{experiment_name}/test.yaml"

    with open(cfg_path) as f:
        cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))


    # Check if experiment_name is provided, otherwise assume it as "baseline"
    if args["experiment_name"] is None:
        args["experiment_name"] = "baseline"
        print(f"{Fore.YELLOW}Missing input 'experiment_name'. Assumed to be 'baseline'{Fore.RESET}")
    
    # Check if identifier is provided, otherwise set it as "1"


    if args["identifier"] == "":
        args["identifier"] = "1"
    elif args["identifier"] == "auto":
        # Find the latest experiment folder and increment the identifier
        files = os.listdir(f'{PHD_RESULTS}/models/{cfg.project}/{experiment_name}')
        folder_experiments = [int(s) for s in files if s.isdigit()]

        if len(folder_experiments) == 0:
            args["identifier"] = 1
        else:
            folder_experiments.sort()
            args["identifier"] = folder_experiments[-1] + 1 
            


    identifier = args['identifier']
    # identifier = f"/{args['identifier']}"

    log_path = f'{PHD_RESULTS}/logs/{cfg.project}/{experiment_name}/{identifier}'


    experiment_path = f'{PHD_RESULTS}/models/{cfg.project}/{experiment_name}/{args["identifier"]}'    
    print(experiment_path)
    if not os.path.exists(experiment_path):
        raise Exception(f"Results from experiment '{experiment_name}' does not exist in path: {experiment_path}")
    
    return log_path, experiment_path, cfg


# Logger function to save training loss to CSV
def log_to_csv(epoch, loss, val_loss, log_file):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file doesn't exist
            writer.writerow(['Epoch', 'Loss', 'Val Loss'])
        # Write the epoch and loss
        writer.writerow([epoch, loss, val_loss])