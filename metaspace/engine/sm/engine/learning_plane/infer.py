from typing import Callable

from lithops.config import load_config

from sm.engine.learning_plane.io.io_model import load_model, infer_read, infer_write
from sm.engine.learning_plane.io.infer.model import IOModel
from sm.engine.learning_plane.cpu import calculate_cpu
from sm.engine.learning_plane.overhead import calculate_overhead

MB = 1024 ** 2

def infer_parent(
    config: dict,
    data_size: int,
    function_name: str,
    p: int,
    read_requests_worker: int,
    write_requests_parent: Callable[[int, int], int],
    read_requests_child: Callable[[int, int], int],
    io_model: IOModel,
    total_files: int = 1) -> float:

    read_time = infer_read(data_size, p, read_requests_worker, data_size / p / MB, total_files, io_model)
    compute_time = calculate_cpu(function_name, data_size / p)
    if compute_time < 0:
        compute_time = 0
    oh_time = calculate_overhead(config, p)
    _write_requests_parent = write_requests_parent(p, p)
    _read_requests_child = read_requests_child(p, p)
    write_time = infer_write(data_size, p, _write_requests_parent, data_size / p / _write_requests_parent / MB, p, io_model)
    child_read_time = infer_read(data_size, p, _read_requests_child, data_size / p / _read_requests_child / MB, p, io_model)
    return read_time + compute_time + oh_time + write_time + child_read_time


def infer_child(
    config: dict,
    data_size: int,
    function_name: str,
    p_parent: int,
    p: int,
    write_requests_parent: Callable[[int, int], int],
    read_requests_child: Callable[[int, int], int],
    write_requests_worker: int,
    io_model: IOModel,
    total_files: int = 1) -> float:

    _write_requests_parent = write_requests_parent(p_parent, p)
    _read_requests_child = read_requests_child(p_parent, p)
    parent_write_time = infer_write(data_size, p_parent, _write_requests_parent, data_size / p_parent / _write_requests_parent / MB, p_parent, io_model)
    read_time = infer_read(data_size, p, _read_requests_child, data_size / p / _read_requests_child / MB, p, io_model)

    compute_time = calculate_cpu(function_name, data_size / p)
    if compute_time < 0:
        compute_time = 0
    oh_time = calculate_overhead(config, p)
    _write_requests_worker = write_requests_parent(p_parent, p)
    _read_requests_child = read_requests_child(p_parent, p)
    write_time = infer_write(data_size, p, 1, data_size / p / MB, p, io_model)
    return parent_write_time + read_time + compute_time + oh_time + write_time


def infer(
    data_size: int,
    stage1: str,
    stage2: str,
    p1_read_requests_worker: int,
    # Write requests as a function of p1 and p2
    p1_write_requests_worker: Callable[[int, int], int],
    # Write requests as a function of p1 and p2
    p2_read_requests_worker: Callable[[int, int], int],
    p2_write_requests_worker: int,
    p1: int = None):

    io_model = load_model()
    config = load_config()

    if p1 is None:
        p_range = range(1, 1000, 1)

        parent_p = -1
        parent_p_time = float("inf")
        for p in p_range:
            parent_time = infer_parent(
                config,
                data_size,
                stage1,
                p,
                p1_read_requests_worker,
                p1_write_requests_worker,
                p2_read_requests_worker,
                io_model
            )
            if parent_time < parent_p_time:
                parent_p_time = parent_time
                parent_p = p
            else:
                break
    else:
        parent_p = p1

    p_range = range(1, 1000, 1)
    child_p = -1
    child_p_time = float("inf")
    for p in p_range:
        child_time = infer_child(
            config,
            data_size,
            stage2,
            parent_p, 
            p,
            p1_write_requests_worker,
            p2_read_requests_worker,
            p2_write_requests_worker,
            io_model
        )
        if child_time < child_p_time:
            child_p_time = child_time
            child_p = p
        else:
            break

    return parent_p, child_p

    

