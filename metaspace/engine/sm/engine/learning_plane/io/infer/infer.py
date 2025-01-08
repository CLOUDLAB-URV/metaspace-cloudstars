from typing import Tuple

from sm.engine.learning_plane.io.infer.model import IOModel

MAX_PARTITION_SIZE = 250
MIN_PARTITION_SIZE = 25
MINIMUM_WORKER_MEMORY_PERC = 0.05
MAXIMUM_WORKER_MEMORY_PERC = 0.25
WORKER_NUMBER_STEP = 1
READ_THROUGHPUT_LIMIT_PREFIX = 5500
WRITE_THROUGHPUT_LIMIT_PREFIX = 3500
AGG_THROUGHPUT_LIMIT = 55000
MB = 1024 * 1024

def get_write_time(D: int,
                   model: IOModel,
                   p: int,
                   requests_per_worker: int = 1,
                   mb_per_file: float = None,
                   total_files: int = None) -> Tuple[int, float]:

    dmb = D / MB
    
    if total_files is None:
        total_files = p

    throughput_per_worker = model.get_throughput_write(p,
                                                       mb_per_file 
                                                       if mb_per_file is not None 
                                                       else dmb / p / requests_per_worker )
    io_time = eq_io(
        dmb,
        p,
        requests_per_worker,
        model.get_write_bandwidth_aggregate(),
        model.get_write_bandwidth_per_worker(),
        throughput_per_worker,
        total_files,
        WRITE_THROUGHPUT_LIMIT_PREFIX,
        AGG_THROUGHPUT_LIMIT
    )

    return io_time


def get_read_time(D: int,
                  model: IOModel,
                  p: int,
                  requests_per_worker: int = 1,
                  mb_per_file: float = None,
                  total_files: int = None) -> Tuple[int, float]:

    dmb = D / MB
    
    if total_files is None:
        total_files = p

    throughput_per_worker = model.get_throughput_read(p,
                                                      mb_per_file
                                                      if mb_per_file is not None
                                                      else dmb / p / requests_per_worker )
    io_time = eq_io(
        dmb,
        p,
        requests_per_worker,
        model.get_read_bandwidth_aggregate(),
        model.get_read_bandwidth_per_worker(),
        throughput_per_worker,
        total_files,
        READ_THROUGHPUT_LIMIT_PREFIX,
        AGG_THROUGHPUT_LIMIT
    )

    return io_time


def eq_io(
    D,
    p,
    requests_per_worker,
    bandwidth_aggregate,
    bandwidth_per_worker,
    throughput_per_worker,
    total_files,
    per_file_throughput,
    max_aggregate_throughput: int = None
    ):
    
    bandwidth_limit = min(bandwidth_per_worker * p, bandwidth_aggregate)
    bandwidth_time = D / bandwidth_limit

    worker_throughput_time = p * requests_per_worker / (throughput_per_worker * p)
    file_throughput_time = p * requests_per_worker / (total_files * per_file_throughput) 
    if max_aggregate_throughput:
        max_throughput_time = ( p * requests_per_worker ) / max_aggregate_throughput
    else:
        max_throughput_time = worker_throughput_time
    throughput_limit = max([worker_throughput_time, file_throughput_time, max_throughput_time]) 

    return max(bandwidth_time, throughput_limit)

