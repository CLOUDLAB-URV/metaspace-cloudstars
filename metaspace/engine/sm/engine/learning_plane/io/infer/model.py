from typing import Callable, List, Tuple, Dict
from functools import partial

import numpy as np
from scipy.optimize import curve_fit

DEGREES = [3, 4, 5, 6]
LINEAR_THRESHOLD = 0.95
MAXIMUM_WORKERS = 10**4
READ_THROUGHPUT_LIMIT = 50
WRITE_THROUGHPUT_LIMIT = 40

percentile = 5

def distribution_throughput(data):
    return np.percentile(data, percentile)


def linear(x, a, b):
    return a * x + b


def decaying(x, a, b):
    return a * np.log(x) + b


def fit_and_calculate_rss(func, x, y):
    params, _ = curve_fit(func, x, y)
    y_pred = func(x, *params)
    rss = np.sum((y - y_pred) ** 2)
    return rss, partial(func, *params), params


class IOModel():
  
    def __init__(self):
        
        self.file_size: int = None
        self.models_read: Dict[int, Callable] = None
        self.models_write: Dict[int, Callable] = None
        self.thresholds_read: List[int] = None
        self.thresholds_write: List[int] = None
        
        self.read_bandwidth_aggregate: float = None
        self.write_bandwidth_aggregate: float = None
        self.read_bandwidth_per_worker: float = None
        self.write_bandwidth_per_worker: float = None
        
        self.read_data: Dict[int, Dict[int, float]] = None
        self.write_data: Dict[int, Dict[int, float]] = None
    

    def gen_models(self,
                   data):
            
        self.get_data_model(data)

        self.models_write = {
            file_size : self._model(self.write_data[file_size])[0]
            for file_size in self.write_data
        }
        self.models_read = {
            file_size : self._model(self.read_data[file_size])[0]
            for file_size in self.read_data
        }

  
    def _model(self, data: Dict[int, float]) -> Tuple[Callable, Dict]:

        x = np.array(list(data.keys()))
        y = np.array(list(data.values()))

        rss_linear, linear_fun, linear_params = fit_and_calculate_rss(linear, x, y)
        rss_decaying, decay_func, decay_params = fit_and_calculate_rss(decaying, x, y)

        if rss_linear < rss_decaying:
            return linear_fun, linear_params
        else:
            return decay_func, decay_params


    def get_write_bandwidth_aggregate(self):

        return self.write_bandwidth_aggregate


    def get_read_bandwidth_aggregate(self):

        return self.read_bandwidth_aggregate


    def get_write_bandwidth_per_worker(self):

        return self.write_bandwidth_per_worker


    def get_read_bandwidth_per_worker(self):

        return self.read_bandwidth_per_worker


    def _get_throughput(self, model, p):

        return model(p)


    def get_throughput(self, 
                       p: int, 
                       file_size: float, 
                       models: Dict[int, Callable],
                       upper_throughput_limit: int = None) -> float:
        
        if upper_throughput_limit is None:
            upper_throughput_limit = READ_THROUGHPUT_LIMIT

        file_sizes = list(self.models_read.keys())
        pos = np.searchsorted(file_sizes, file_size)

        if pos == len(file_sizes):
            file_size = file_sizes[-1]
            throughput = self._get_throughput(models[file_size], p)
            return throughput
        elif pos == 0:
            file_size = list(self.models_read.keys())[pos]
            throughput = self._get_throughput(models[file_size], p)
            return throughput     
        else:
            lower_file_size = list(self.models_read.keys())[pos-1]
            upper_file_size = list(self.models_read.keys())[pos]
            lower_throughput = self._get_throughput(models[upper_file_size], p)
            upper_throughput = self._get_throughput(models[lower_file_size], p)


        # Interpolate throughput for the given file_size
        m = (lower_throughput - upper_throughput) / (upper_file_size - lower_file_size)
        n = lower_throughput - m * upper_file_size

        return m * file_size + n


    def get_throughput_read(self, p: int, file_size: float = None):

        if file_size is None:
            file_size = list(self.models_read.keys())[0]

        return self.get_throughput(p, file_size, self.models_read, READ_THROUGHPUT_LIMIT)
    

    def get_throughput_write(self, p: int, file_size: float = None):

        if file_size is None:
            file_size = list(self.models_write.keys())[0]

        return self.get_throughput(p, file_size, self.models_write, WRITE_THROUGHPUT_LIMIT)

    
    def get_data_model(self,
                       data: Dict) -> Tuple[Dict[int, float], Dict[int, float]]:
        
        """
        Calculates the average write and read data for each number of workers based on the given data.

        Args:
            data (Dict): A dictionary containing storages samples in the following format:
                {
                    'samples': [
                        {
                            'workers': int,
                            'write': float,
                            'read': float,
                            'file_size': int
                        },
                        ...
                    ]
                }
        """

        throughput_agg_function = distribution_throughput

        write_throughput_per_worker = dict()
        read_throughput_per_worker = dict()
        
        write_bandwidth_aggregate = []
        read_bandwidth_aggregate = []

        write_bandwidth_per_worker = []
        read_bandwidth_per_worker = []
        
        for s in data['samples']:
            
            file_size = s["file_size"]
            
            if file_size not in write_throughput_per_worker.keys():
                write_throughput_per_worker[file_size] = dict()
                read_throughput_per_worker[file_size] = dict()
            
            w = s["workers"]
            
            if s["workers"] not in write_throughput_per_worker[file_size].keys():
                write_throughput_per_worker[file_size][w] = []
                read_throughput_per_worker[file_size][w] = []
                
            write_throughput_per_worker[file_size][w].extend(s["write"])
            read_throughput_per_worker[file_size][w].extend(s["read"])
            write_bandwidth_aggregate.append(sum(s["write_bandwidth"]))
            read_bandwidth_aggregate.append(sum(s["read_bandwidth"]))
            write_bandwidth_per_worker.extend(s["write_bandwidth"])
            read_bandwidth_per_worker.extend(s["read_bandwidth"])
    
        self.write_data = dict()
        self.read_data = dict()

        for file_size in write_throughput_per_worker:
            
            write_throughput_per_worker[file_size] = {w: throughput_agg_function(write_throughput_per_worker[file_size][w]) 
                                                      for w in write_throughput_per_worker[file_size].keys()}
            self.write_data[file_size] = dict(sorted(write_throughput_per_worker[file_size].items()))
            
            read_throughput_per_worker[file_size] = {w: throughput_agg_function(read_throughput_per_worker[file_size][w])
                                                     for w in read_throughput_per_worker[file_size].keys()}
            self.read_data[file_size] = dict(sorted(read_throughput_per_worker[file_size].items()))
            
            self.read_bandwidth_aggregate = max(read_bandwidth_aggregate)
            self.write_bandwidth_aggregate = max(write_bandwidth_aggregate)
            self.read_bandwidth_per_worker = max(read_bandwidth_per_worker)
            self.write_bandwidth_per_worker = max(write_bandwidth_per_worker)

        self.read_data = dict(sorted(self.read_data.items()))
        self.write_data = dict(sorted(self.write_data.items()))

