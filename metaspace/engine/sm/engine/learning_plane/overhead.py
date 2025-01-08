from typing import TypedDict, Dict

class OverheadFunction(TypedDict):
    a: float
    b: float
    c: float


# X: partition size
def oh_func(x, a, b, c):
    return max(0.01, a * x ** 2 + b * x + c)

REGISTERED_BACKENDS: Dict[str, OverheadFunction] = {
    "localhost": { "a": 0, "b": 0, "c": 0 },
    "aws_lambda": {
        "us-east-1": { "a": 0.00001059259, "b": -0.002210752, "c": 1.923945 },
    }
}

def calculate_overhead(lithops_config: dict, x: int):

    backend = lithops_config["lithops"]["backend"]
    args = REGISTERED_BACKENDS[backend].values()
    if backend != "localhost":
        if backend.startswith("aws"):
            region = lithops_config["aws"]["region"]
            args = REGISTERED_BACKENDS[backend][region].values()

    return oh_func(x, *args)


