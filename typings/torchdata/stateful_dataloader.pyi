from _typeshed import Incomplete
from torch.utils.data import DataLoader

class StatefulDataLoader(DataLoader):
    dataset: Incomplete
    batch_size: int | None
    shuffle: bool
    sampler: Incomplete
    batch_sampler: Incomplete
    num_workers: int
    collate_fn: Incomplete
    pin_memory: bool
    drop_last: bool
    timeout: float
    worker_init_fn: Incomplete
    multiprocessing_context: Incomplete
    generator: Incomplete
    prefetch_factor: int
    persistent_workers: bool
    pin_memory_device: str

    def __init__(
        self,
        dataset,
        batch_size: int | None = None,
        shuffle: bool = False,
        sampler=None,
        batch_sampler=None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
