import numpy as np
from typing import TypedDict, Dict

class CPUFunction(TypedDict):
    a: float
    b: float
    c: float
    d: float

# X: partition size
def comp_func(x, a, b, c, d):
    return max(0.01, a / x + b * np.log(x) / x + c / x**2 + d)

REGISTERED_STAGES: Dict[str, CPUFunction] = {
    "upload_partitions": { "a": 45.257704034166395, "b": -19319713.344141785, "c": 1.0, "d": 15.41083745932142 },
    "merge_segment": { "a": 7620957643.830114, "b": -448705556.16195744, "c": 1.0, "d": 9.16015346280316 }
}

def calculate_cpu(stage_name: str, x: int):
    if stage_name not in REGISTERED_STAGES:
        raise ValueError(f"Unknown stage {stage_name}")
    return comp_func(x, *REGISTERED_STAGES[stage_name].values())
