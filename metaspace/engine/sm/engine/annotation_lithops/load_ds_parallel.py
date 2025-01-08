from concurrent.futures import ThreadPoolExecutor
import time
import logging
from typing import Union

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from lithops.storage import Storage

from sm.engine.ds_config import DSConfig, DSPartitioningMode
from sm.engine.annotation.imzml_reader import LithopsImzMLReader
from sm.engine.annotation_lithops.executor import SINGLE_CPU_MEMORY_MB, Executor
from sm.engine.annotation_lithops.io import load_cobj, save_cobj, CloudObject
from sm.engine.learning_plane.infer import infer


logger = logging.getLogger('annotation-pipeline')

MAX_CHUNK_SIZE = 128 * 1024 ** 2  # 128MB
MAX_MZ_VALUE = 10 ** 5


def get_expected_size(imzml_reader: LithopsImzMLReader):
    row_size = sum([4, np.dtype(imzml_reader.imzml_reader.mzPrecision).itemsize, np.dtype(imzml_reader.imzml_reader.intensityPrecision).itemsize])
    return sum([ imzml_reader.imzml_reader.mzLengths[sp_i] * row_size for sp_i in np.argsort(imzml_reader.imzml_reader.intensityOffsets) ])


def define_ds_segments(executor: Executor,
                       imzml_cobject: CloudObject,
                       ibd_cobject: CloudObject,
                       imzml_reader: LithopsImzMLReader,
                       sample_n: int,
                       segm_n: int = None,
                       ds_segm_size_mb: int = None):

    def get_segm_bounds(segm_n, storage: Storage):
        
        sp_n = len(imzml_reader.imzml_reader.coordinates)
        sample_sp_inds = np.random.choice(np.arange(sp_n), min(sp_n, sample_n))
        print(f'Sampling {len(sample_sp_inds)} spectra')

        spectra_sample = list(imzml_reader.iter_spectra(storage,
                                                        sample_sp_inds))

        spectra_mzs = np.concatenate([mzs for _, mzs, _ in spectra_sample])
        print(f'Got {len(spectra_mzs)} mzs')

        if segm_n is None:
            total_size = 3 * spectra_mzs.nbytes * sp_n / len(sample_sp_inds)
            segm_n = int(np.ceil(total_size / (ds_segm_size_mb * 2 ** 20)))

        segm_bounds_q = [i * 1 / segm_n for i in range(0, segm_n + 1)]
        segm_lower_bounds = [np.quantile(spectra_mzs, q) for q in segm_bounds_q]
        ds_segm_bounds = np.array(list(zip(segm_lower_bounds[:-1], segm_lower_bounds[1:])))

        # extend boundaries of the first and last segments
        # to include all mzs outside of the spectra sample mz range
        ds_segm_bounds[0, 0] = 0
        ds_segm_bounds[-1, 1] = MAX_MZ_VALUE

        return ds_segm_bounds

    logger.info('Defining dataset segments bounds')
    memory_capacity_mb = 1024
    ds_segm_bounds = executor.call(
        get_segm_bounds,
        (segm_n),
        runtime_memory=memory_capacity_mb,
    )
    return ds_segm_bounds


def partition_spectra(executor: Executor,
                  ibd_cobject: CloudObject,
                  imzml_cobject: CloudObject,
                  ds_segm_bounds: np.ndarray,
                  num_chunks: int,
                  imzml_reader: LithopsImzMLReader):
            
    # sp_id_to_idx = get_pixel_indices(imzml_reader.coordinates)
    row_size = sum([4,
                    np.dtype(imzml_reader.imzml_reader.mzPrecision).itemsize,
                    np.dtype(imzml_reader.imzml_reader.intensityPrecision).itemsize])

    # define first level segmentation and then segment each one into desired number

    def plan_chunks(num_chunks: int = None):

        if num_chunks is not None:
            chunk_size = len(imzml_reader.imzml_reader.intensityOffsets) // num_chunks
            for i in range(num_chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i != num_chunks - 1 else len(imzml_reader.imzml_reader.intensityOffsets)
                yield (start, end)
        else:
            chunk_sp_inds = []
            estimated_size_mb = 0
            # Iterate in the same order that intensities are laid out in the file, hopefully this will
            # prevent fragmented read patterns
            for sp_i in np.argsort(imzml_reader.imzml_reader.intensityOffsets):
                spectrum_size = imzml_reader.imzml_reader.mzLengths[sp_i] * row_size
                if estimated_size_mb + spectrum_size > MAX_CHUNK_SIZE:
                    estimated_size_mb = 0
                    yield np.array(chunk_sp_inds)
                    chunk_sp_inds = []

                estimated_size_mb += spectrum_size
                chunk_sp_inds.append(sp_i)

            if chunk_sp_inds:
                yield np.array(chunk_sp_inds)

    def upload_partitions(ch_i, storage):
        
        ch = chunks[ch_i]

        if len(ch) == 2:
            sp_start = ch[0]
            sp_end = ch[1]
            sp_inds = range(sp_start, sp_end)
        else:
            sp_inds = ch

        mz_arrays = [np.array([], dtype=imzml_reader.mz_precision)] * imzml_reader.n_spectra
        int_arrays = [np.array([], dtype=np.float32)] * imzml_reader.n_spectra
        sp_lens = np.zeros(len(sp_inds), np.int64)
        base_sp_i = sp_inds[0]

        for sp_i, mzs, ints in imzml_reader.iter_spectra(storage, sp_inds):
            mz_arrays[sp_i] = mzs
            int_arrays[sp_i] = ints.astype(np.float32)
            sp_lens[sp_i-base_sp_i] = len(ints)

        ints = np.concatenate(int_arrays)

        mzs = np.concatenate(mz_arrays)

        sp_idxs = np.empty(len(ints), np.uint32)
        sp_lens = np.insert(np.cumsum(sp_lens), 0, 0)
        for sp_ii, sp_i in enumerate(sp_inds):
            start = sp_lens[sp_ii]
            end = sp_lens[sp_ii + 1]
            sp_idxs[start:end] = imzml_reader.pixel_indexes[sp_i]

        dest_bounds = ds_segm_bounds[:, 1]
        destination_indexes = np.searchsorted(dest_bounds, mzs)

        chunk = pd.DataFrame({
            'mz': mzs,
            'int': ints,
            'sp_i': sp_idxs,
        },
        index=pd.RangeIndex(0, len(mzs)))

        def _intermediate_partition_upload(segm_i):
            partition = chunk.loc[destination_indexes == segm_i]
            retry = 0
            while True:
                try:
                    return save_cobj(storage, partition)
                except ClientError as e:
                    retry += 1
                    if retry > 5:
                        raise e
                    else:
                        time.sleep(3)

        with ThreadPoolExecutor(max_workers=len(ds_segm_bounds)) as pool:
            sub_segms_cobjects = list(pool.map(_intermediate_partition_upload,
                                               range(len(ds_segm_bounds))))

        return sub_segms_cobjects

    chunks = list(plan_chunks(num_chunks))
    memory_capacity_mb = SINGLE_CPU_MEMORY_MB
    ds_chunks_cobjects= executor.map_unpack(
        upload_partitions,
        [(i,) for i in range(len(chunks))],
        runtime_memory=memory_capacity_mb
    )
    
    return ds_chunks_cobjects


def merge_spectra_segments(executor: Executor,
                           partition_cobjects,
                           ibd_cobject: CloudObject,
                           imzml_cobject: CloudObject,
                           ds_segm_bounds: np.ndarray):
                           

    def merge_segment(segm_cobjects,
                      storage):
  
        print(f'Merging segment {id} spectra chunks')

        def load_segment(ch_i):
            segm_spectra_chunk = load_cobj(storage, segm_cobjects[ch_i])
            return segm_spectra_chunk

        with ThreadPoolExecutor(max_workers=len(segm_cobjects)) as pool:
            segm = list(pool.map(load_segment, range(len(segm_cobjects))))

        segm = pd.concat(segm, ignore_index=True)

        # Alternative in-place sorting (slower) :
        segm.sort_values(by='mz', inplace=True, kind = "mergesort")
        segm.reset_index(drop=True, inplace=True)

        segm_cobj = save_cobj(storage, segm)

        return segm_cobj, len(segm)

    second_level_segms_cobjects = np.transpose(partition_cobjects).tolist()
    second_level_segms_cobjects = [(segm_cobjects,) for segm_cobjects in second_level_segms_cobjects]

    ds_segms_cobjects, ds_segms_lens = executor.map_unpack(
        merge_segment,
        second_level_segms_cobjects,
        runtime_memory=SINGLE_CPU_MEMORY_MB
    )

    for cobjects in partition_cobjects:
        executor.storage.delete_cloudobjects(cobjects)

    return ds_segms_cobjects, ds_segms_lens


def set_load_ds_parallelism(ds_config: Union[dict, DSConfig], imzml_reader: LithopsImzMLReader):

    num_chunks = None
    segm_n = None

    if "partitioning_mode" in ds_config:
        partitioning_mode = DSPartitioningMode(ds_config["partitioning_mode"])
    else:
        partitioning_mode = DSPartitioningMode.PARTITION_SIZE
    
    if partitioning_mode == DSPartitioningMode.PARAM:
        num_chunks = ds_config.get("partition_number")
        segm_n = ds_config.get("segment_number")
        if num_chunks is None:
            num_chunks = segm_n
        elif segm_n is None:
            segm_n = num_chunks
        if num_chunks is None:
            logger.warning('Both partition_number and segment_number are None, running on PARTITION_SIZE mode')
            partitioning_mode = DSPartitioningMode.PARTITION_SIZE
    elif partitioning_mode == DSPartitioningMode.SMART:
        expected_size = get_expected_size(imzml_reader)
        try:
            num_chunks, segm_n = infer(
                expected_size,
                "upload_partitions",
                "merge_segment",
                1,
                lambda p1, p2: 1,
                lambda p1, p2: p1,
                1
            )
        except FileNotFoundError as e:
            print(e)
            logger.warning('Could not infer partitioning parameters, running on PARTITION_SIZE mode')

    return num_chunks, segm_n

def load_ds_parallel(
        executor: Executor,
        imzml_cobject: CloudObject,
        ibd_cobject: CloudObject,
        ds_segm_size_mb: int,
        ds_config: Union[dict, DSConfig]
):

    sample_sp_n = 500

    storage = executor.storage
    imzml_reader = LithopsImzMLReader(storage, imzml_cobject, ibd_cobject)

    num_chunks, segm_n = set_load_ds_parallelism(ds_config, imzml_reader)

    ds_segm_bounds = define_ds_segments(executor,
                                        imzml_cobject,
                                        ibd_cobject,
                                        imzml_reader,
                                        sample_sp_n,
                                        segm_n,
                                        ds_segm_size_mb)
    
    segm_n = len(ds_segm_bounds)
    
    ds_chunks = partition_spectra(executor,
                                  ibd_cobject,
                                  imzml_cobject,
                                  ds_segm_bounds,
                                  num_chunks,
                                  imzml_reader)
    
    ds_segm_cobjects, ds_segm_lens = merge_spectra_segments(executor,
                                                            ds_chunks,
                                                            ibd_cobject,
                                                            imzml_cobject,
                                                            ds_segm_bounds)
    

    return imzml_reader, ds_segm_bounds, ds_segm_cobjects, ds_segm_lens
    

