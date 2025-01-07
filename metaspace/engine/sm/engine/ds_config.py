from typing import TypedDict, List, Optional
from enum import Enum


class DSConfigIsotopeGeneration(TypedDict):
    adducts: List[str]
    charge: int
    isocalc_sigma: float
    instrument: str
    n_peaks: int
    neutral_losses: List[str]
    chem_mods: List[str]


class DSConfigFDR(TypedDict):
    decoy_sample_size: int
    scoring_model_id: Optional[int]


class DSConfigImageGeneration(TypedDict):
    ppm: float
    n_levels: int
    min_px: int
    # Disables an optimization where expensive metrics are skipped if cheap metrics already indicate
    # the annotation will be rejected. Only useful for collecting data for model training.
    compute_unused_metrics: Optional[bool]


class DSPartitioningMode(Enum):
    PARAM = 1
    PARTITION_SIZE = 2
    SMART = 3

    def toJSON(self):
        return self.value  # or self.value


class DSConfig(TypedDict):
    database_ids: List[int]
    ontology_db_ids: List[int]
    analysis_version: int
    isotope_generation: DSConfigIsotopeGeneration
    fdr: DSConfigFDR
    image_generation: DSConfigImageGeneration
    parallel_load_ds: bool
    partitioning_mode: DSPartitioningMode
    partition_number: int
    segment_number: int
