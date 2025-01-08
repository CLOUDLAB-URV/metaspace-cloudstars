import os
import datetime

import pickle
from lithops.config import load_config


BASE_DIR = "/opt/dev/metaspace/metaspace/engine/learning_plane"
HOME_DIR = os.path.expanduser("~") + "/metaspace/learning_plane"

def get_learning_plane_dir():
    if os.path.exists(BASE_DIR):
        if os.access(BASE_DIR, os.R_OK | os.W_OK):
            return BASE_DIR
    else:
        try:
            os.makedirs(BASE_DIR)
            return BASE_DIR
        except PermissionError:
            pass

    if os.path.exists(HOME_DIR):
        if os.access(HOME_DIR, os.R_OK | os.W_OK):
            return HOME_DIR
    else:
        try:
            os.makedirs(HOME_DIR)
            return HOME_DIR
        except PermissionError:
            pass
    raise PermissionError("No suitable directory found to save model.")

def get_data_path(storage_backend: str):
    data_path = os.path.join(get_learning_plane_dir(), "data", storage_backend)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path

def get_throughput(data: dict):
    return data["ops"]


def get_bandwidth(data: dict, begin_datetime: datetime.datetime, file_size: int):
    if "first_write_timestamp" in data.keys():
        end_datetime = data["first_write_timestamp"]
    else:
        end_datetime = data["first_read_timestamp"]
    
    if end_datetime is None:
        return 0
    else:
        time = (end_datetime - begin_datetime).total_seconds()
        bandwidth = file_size / time
        return bandwidth


def read_samples(dir: str = None):
    
    storage = load_config()['lithops']['storage']
    if dir is None:
        dir = os.path.join(get_learning_plane_dir(), storage)
    
    samples = {"samples": [], "date": datetime.date.today().strftime("%Y-%m-%d")}

    files = []
    for file in os.listdir(dir):
        
        if file.endswith("write.pickle"):
            files.append(os.path.join(dir, file))
            

    for file in files:
        
        write_file = file
        read_file = file.replace("write", "read")
        
        write_data = pickle.load(open(write_file, "rb"))
        workers = len(write_data["results"])
        file_size = write_data["mb_per_file"]
        
        write_throughput = [ get_throughput(r) for r in write_data["results"] ]
    
        worker_write_bandwidth = [ get_bandwidth(d, write_data["start_time"], write_data["mb_per_file"]) for d in write_data["results"] ]
        
        read_data = pickle.load(open(read_file, "rb"))
        read_throughput = [ get_throughput(r) for r in read_data["results"] ]

        worker_read_bandwidth = [ get_bandwidth(d, read_data["start_time"], read_data["mb_per_file"]) for d in read_data["results"] ]
        
        samples["samples"].append({"workers": workers, 
                                   "write": write_throughput,
                                   "read": read_throughput,
                                   "write_bandwidth": worker_write_bandwidth,
                                   "read_bandwidth": worker_read_bandwidth,
                                   "file_size": file_size})

    return samples

