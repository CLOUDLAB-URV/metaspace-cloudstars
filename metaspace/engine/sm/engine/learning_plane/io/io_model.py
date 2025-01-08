import os
from typing import List

import pickle
from lithops.config import load_config

from sm.engine.config import SMConfig
from sm.engine.annotation_lithops.executor import SINGLE_CPU_MEMORY_MB
from sm.engine.learning_plane.io.profile.profile import profile
from sm.engine.learning_plane.io.infer.data import get_learning_plane_dir, read_samples, get_data_path
from sm.engine.learning_plane.io.infer.infer import get_read_time, get_write_time
from sm.engine.learning_plane.io.infer.model import IOModel

DEFAULT_PROFILE_WORKERS = [ 200, 400, 600, 800, 1000 ]
DEFAULT_FILE_SIZES =  [ 5, 25 ]


def get_storage_backend():
    return load_config()['lithops']['storage']

def get_model_path(storage_backend: str) -> str:

    model_dir = os.path.join(get_learning_plane_dir(), "models", storage_backend)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"model.pickle")
    return model_path

def check_model_path(storage_backend: str) -> bool:

    model_path = get_model_path(storage_backend)
    
    if os.path.exists(model_path):
        print(f"Model for backend {storage_backend} already exists.")
        print("Do you still want to profile the storage?")
        user_input = input("(Y/n): ")
        if user_input.lower() == "Y":
            return True
        else:
            return False
    else:
        return True
     

def setup_storage_profile(
        bucket_name: str = None,
        workers: List[int] = DEFAULT_PROFILE_WORKERS,
        file_sizes: List[int] = DEFAULT_FILE_SIZES):
            
    storage_backend = get_storage_backend()
    lithops_config = load_config()

    if bucket_name is None:
        if storage_backend in lithops_config and "storage_bucket" in lithops_config[storage_backend]:
            bucket_name = lithops_config[storage_backend]["storage_bucket"]
            
    if bucket_name is None:
        raise ValueError("Bucket name is not provided and it is not found in the config file")
    print("Performing setup for backend: ", storage_backend, " against bucket: ", bucket_name)    
            
    if not check_model_path(storage_backend):
        return
    
    profile_storage(bucket_name,
            workers,
            file_sizes)
    

def profile_storage(
        bucket_name: str = None,
        workers: List[int] = DEFAULT_PROFILE_WORKERS,
        file_sizes: List[int] = DEFAULT_FILE_SIZES):
    if bucket_name is None:
        config = SMConfig.get_conf()
        bucket_name = config["lithops"]["sm_storage"]["pipeline_cache"][0]

    for file_size in file_sizes:
        profile(bucket_name=bucket_name,
                mb_per_file=file_size,
                functions = workers,
                runtime_memory = SINGLE_CPU_MEMORY_MB,
                replica_number = 3)

        
def create_io_model(save: bool = True,
                    data_path: str = None) -> IOModel:
    storage_backend = get_storage_backend()
    if data_path is None:
        if not check_model_path(storage_backend):
            return
        else:
            data_path = get_data_path(storage_backend)

    samples = read_samples(data_path)
    model = IOModel()
    model.gen_models(samples)

    if save:
        model_path = get_model_path(storage_backend)
        pickle.dump(model, open(model_path, "wb"))
        print(f"Model saved at {model_path}.")

    return model


def load_model():
    storage_backend = get_storage_backend()
    model_path = get_model_path(storage_backend)
    if not os.path.exists(model_path):
        data_path = get_data_path(storage_backend)
        if not os.path.exists(data_path):
            raise FileNotFoundError("No model nor data to generate the model.")
        else:
            return create_io_model(save = True, data_path = data_path)
    else:
        return pickle.load(open(model_path, "rb"))

    
def infer_write(D: int,
                p: int = None,
                requests_per_worker: int = 1,
                mb_per_file: float = None,
                total_files: int = None,
                model: IOModel = None,
                model_path: str = None):

    if model is None:
        if model_path is None:
            model = load_model()
        else:
            model = pickle.load(open(model_path, "rb"))

    return get_write_time(D = D, 
                          model = model, 
                          p = p,
                          requests_per_worker=requests_per_worker, 
                          mb_per_file=mb_per_file,
                          total_files=total_files)
    
def infer_read(D: int,
                p: int = None,
                requests_per_worker: int = 1,
                mb_per_file: float = None,
                total_files: int = None,
                model: IOModel = None,
                model_path: str = None):

    if model is None:
        if model_path is None:
            model = load_model()
        else:
            model = pickle.load(open(model_path, "rb"))

    return get_read_time(D = D, 
                        model = model, 
                        p = p,
                        requests_per_worker=requests_per_worker, 
                        mb_per_file=mb_per_file,
                        total_files=total_files)
