import time
import datetime
from typing import List, Union
import uuid

import pickle
import numpy as np
from lithops import FunctionExecutor, Storage

from sm.engine.annotation_lithops.executor import SINGLE_CPU_MEMORY_MB
from sm.engine.config import SMConfig
from sm.engine.learning_plane.io.infer.data import get_data_path


DATA_PER_WORKER = 200 * 1024**2
WAIT_INTERVAL = 8
IO_INTERVAL = 3

class RandomDataGenerator(object):
    """
    A file-like object which generates random data.
    1. Never actually keeps all the data in memory so
    can be used to generate huge files.
    2. Actually generates random data to eliminate
    false metrics based on compression.

    It does this by generating data in 1MB blocks
    from np.random where each block is seeded with
    the block number.
    """

    def __init__(self, bytes_total):
        self.bytes_total = bytes_total
        self.pos = 0
        self.current_block_id = None
        self.current_block_data = ""
        self.BLOCK_SIZE_BYTES = 1024*1024
        self.block_random = np.random.randint(0, 256, dtype=np.uint8,
                                              size=self.BLOCK_SIZE_BYTES)

    def __len__(self):
        return self.bytes_total

    @property
    def len(self):
        return self.bytes_total 

    def tell(self):
        return self.pos

    def seek(self, pos, whence=0):
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        elif whence == 2:
            self.pos = self.bytes_total - pos

    def get_block(self, block_id):
        if block_id == self.current_block_id:
            return self.current_block_data

        self.current_block_id = block_id
        self.current_block_data = (block_id + self.block_random).tostring()
        return self.current_block_data

    def get_block_coords(self, abs_pos):
        block_id = abs_pos // self.BLOCK_SIZE_BYTES
        within_block_pos = abs_pos - block_id * self.BLOCK_SIZE_BYTES
        return block_id, within_block_pos

    def read(self, bytes_requested):
        remaining_bytes = self.bytes_total - self.pos
        if remaining_bytes == 0:
            return b''

        bytes_out = min(remaining_bytes, bytes_requested)
        start_pos = self.pos

        byte_data = b''
        byte_pos = 0
        while byte_pos < bytes_out:
            abs_pos = start_pos + byte_pos
            bytes_remaining = bytes_out - byte_pos

            block_id, within_block_pos = self.get_block_coords(abs_pos)
            block = self.get_block(block_id)
            # how many bytes can we copy?
            chunk = block[within_block_pos:within_block_pos + bytes_remaining]

            byte_data += chunk

            byte_pos += len(chunk)

        self.pos += bytes_out

        if self.pos == self.bytes_total:
            self.pos = 0

        return byte_data


def synchro_simple(synchro_time: datetime.datetime, synchro_timezone):
    
    local_datetime = datetime.datetime.now()
    local_datetime = local_datetime.astimezone(synchro_timezone)
    while local_datetime < synchro_time:
        local_datetime = datetime.datetime.now()
        local_datetime = local_datetime.astimezone(synchro_timezone)
        time.sleep(0.01)



def write(bucket_name: str,
          mb_per_file: int,
          number_of_functions: int,
          runtime_memory: int = SINGLE_CPU_MEMORY_MB,
          debug: bool = False):    

    log_level = 'INFO' if not debug else 'DEBUG'
    config = SMConfig.get_conf()
    lithops_config = config["lithops"]
    fexec = FunctionExecutor(
        config=lithops_config,
        runtime_memory=runtime_memory,
        log_level=log_level)
    
    begin_datetime = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=WAIT_INTERVAL)
    end_datetime = begin_datetime + datetime.timedelta(seconds=IO_INTERVAL)

    def writer(func_i: int):
        
        storage = Storage()
        
        bytes_n = mb_per_file * 1024**2
        generator = RandomDataGenerator(bytes_n)
        object = generator.read(bytes_n)

        key_name = "%d%d%d%d"%(func_i, func_i, func_i, func_i) + str(uuid.uuid4().hex.upper())
        
        synchro_simple(begin_datetime, begin_datetime.tzinfo)
        
        write_timestamps = []
        key_names = []
        while True:
            
            storage.put_object(bucket_name, key_name, object)
            write_timestamp = datetime.datetime.now().astimezone()
            if write_timestamp < end_datetime:
                write_timestamps.append(write_timestamp)
                key_names.append(key_name)
            else:
                break

        ops = len(write_timestamps)/(end_datetime - begin_datetime).total_seconds()
        print('WOP/s : '+str(ops))

        if len(write_timestamps) > 0:
            first_write_timestamp = write_timestamps[0]
        else:
            first_write_timestamp = None

        return {'ops': ops,
                'keynames': key_names,
                'write_timestamps': write_timestamps,
                'first_write_timestamp': first_write_timestamp,
                'bytes': bytes_n*len(key_names) }
    
    btime = datetime.datetime.now().astimezone()
    print(f"Starting write: {btime}")
    worker_futures = fexec.map(writer, range(number_of_functions))
    results = fexec.get_result(throw_except=False)
    atime = datetime.datetime.now().astimezone()
    print(f"Ended write: {atime} - ({(atime-btime).total_seconds()})")
    
    results = [gbs for gbs in results if gbs is not None]
    worker_stats = [f.stats for f in worker_futures if not f.error]

    res = {'start_time': begin_datetime,
           'end_time': end_datetime,
           'worker_stats': worker_stats,
           'bucket_name': bucket_name,
           'keynames': [ r["keynames"] for r in results ],
           'results': results,
           "runtime_memory": runtime_memory,
           "mb_per_file": mb_per_file}

    return res


def read(bucket_name: str,
        keylist: List[List[str]],
        mb_per_file: int, 
        runtime_memory: int = 1769, 
        debug: bool = False):
    
    # Ensure no element in keylist has length 0
    non_empty_keys = [keys for keys in keylist if len(keys) > 0]
    if not non_empty_keys:
        raise ValueError("All elements in keylist are empty")

    for i, keys in enumerate(keylist):
        if len(keys) == 0:
            key_i = np.random.randint(0, len(non_empty_keys))
            keylist[i] = non_empty_keys[key_i]
    
    
    blocksize = 1024*1024
    
    log_level = 'INFO' if not debug else 'DEBUG'
    config = SMConfig.get_conf()
    lithops_config = config["lithops"]
    fexec = FunctionExecutor(
        config=lithops_config,
        runtime_memory=runtime_memory,
        log_level=log_level)

    begin_datetime = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=WAIT_INTERVAL)
    end_datetime = begin_datetime + datetime.timedelta(seconds=IO_INTERVAL)

    def reader(keys: List[str]):
        
        storage = Storage()
        bytes_read = 0
        
        synchro_simple(begin_datetime, begin_datetime.tzinfo)

        read_timestamps = []
        key_id = 0

        while True:
            key_name = keys[key_id]

            fileobj = storage.get_object(bucket_name, key_name, stream=True)
            
            try:
                buf = fileobj.read(blocksize)
                
                while len(buf) > 0:
                    bytes_read += len(buf)
                    buf = fileobj.read(blocksize)
            except Exception as e:
                print(e)
                pass

            read_timestamp = datetime.datetime.now().astimezone()

            if read_timestamp < end_datetime:
                read_timestamps.append(read_timestamp)
                key_id = (key_id + 1) % len(keys)
            else:
                break

        ops = len(read_timestamps) / (end_datetime - begin_datetime).total_seconds()

        if len(read_timestamps) > 0:
            first_read_timestamp = read_timestamps[0]
        else:
            first_read_timestamp = None

        return {'ops': ops,
                'read_timestamps': read_timestamps,
                'first_read_timestamp': first_read_timestamp,
                'bytes': bytes_read}

    
    worker_futures = fexec.map(reader, keylist)
    results = fexec.get_result(throw_except=False)

    results = [gbs for gbs in results if gbs is not None]
    worker_stats = [f.stats for f in worker_futures if not f.error]

    res = {
        'start_time': begin_datetime,
        'end_time': end_datetime,
        'worker_stats': worker_stats,
        'results': results,
        "runtime_memory": runtime_memory,
        "mb_per_file": mb_per_file
   }

    return res


def warm_up(runtime_memory: int, 
            number_of_functions: int,
            runtime: str = None):

    def foo(x):
        return x

    config = SMConfig.get_conf()
    lithops_config = config["lithops"]
    fexec = FunctionExecutor(
        config=lithops_config,
        runtime_memory=runtime_memory)
    worker_futures = fexec.map(foo, range(number_of_functions))
    results = fexec.get_result(worker_futures, throw_except=False)


def delete_temp_data(bucket_name: str, 
                     keynames: Union[List[str], List[List[str]]],):
    
    if isinstance(keynames[0], list):
        keynames = [ key for keynames in keynames for key in keynames ]
        
    print('Deleting temp files...')
    storage = Storage()
    try:
        storage.delete_objects(bucket_name, keynames)
    except Exception as e:
        print(e)
        pass
    print('Done!')


def delete_command(key_file: str, 
                   outdir: str, 
                   name: str):
    
    if key_file:
        res_write = pickle.load(open(key_file, 'rb'))
    else:
        res_write = pickle.load(open('{}/{}_write.pickle'.format(outdir, name), 'rb'))
        
    bucket_name = res_write['bucket_name']
    keynames = [ key for keynames in res_write['keynames'] for key in keynames ]
    
    delete_temp_data(bucket_name, keynames)


def _profile(bucket_name: str, 
             mb_per_file: int, 
             number_of_functions: int,
             runtime_memory: int = 1769,
             debug: bool = False, 
             replica_number: int = 1):
    
    config = SMConfig.get_conf()
    lithops_config = config["lithops"]
    storage_backend = lithops_config["lithops"]["storage"]
    date = datetime.datetime.now().strftime("%d-%m-%Y")

    for r_n in range(replica_number):
    
        fname = f"{number_of_functions}_{mb_per_file}_{date}_{r_n}"

        print('Executing Write Test:')
        
        res_write = write(bucket_name,
                          mb_per_file,
                          number_of_functions,
                          runtime_memory,
                          debug)
        
        pickle.dump(res_write, open(f'{get_data_path(storage_backend)}/{fname}_write.pickle', 'wb'), -1)
        print('Sleeping 4 seconds...')
        time.sleep(4)
        print('Executing Read Test:')

        res_read = read(bucket_name,
                        res_write["keynames"],
                        mb_per_file,
                        runtime_memory,
                        debug)
        pickle.dump(res_read, open(f'{get_data_path(storage_backend)}/{fname}_read.pickle', 'wb'), -1)

        delete_temp_data(bucket_name, res_write["keynames"])
    
    
def profile(bucket_name: str,
             mb_per_file: int,
             functions: List[int],
             runtime_memory: int = 1769,
             debug: bool = False,
             replica_number: int = 1):
    
    sorted_functions = sorted(functions, reverse=True)
    print(f"Warming up {functions[0]} functions")
    warm_up(runtime_memory,
            max(functions))
    
    for num_functions in sorted_functions:
        
        print(f"Profiling {num_functions} functions")
        _profile(bucket_name,
                 mb_per_file,
                 num_functions,
                 runtime_memory,
                 debug,
                 replica_number)