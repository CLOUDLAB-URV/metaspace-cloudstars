
DEFAULT_PARTITIONS = 10

def infer(data_size: int, stage1: str, stage2: str, p1: int = None):
    p1 = DEFAULT_PARTITIONS
    p2 = DEFAULT_PARTITIONS

    return p1, p2