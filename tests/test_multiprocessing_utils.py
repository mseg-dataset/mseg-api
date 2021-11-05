"""
Brief multiprocessing example
"""

from mseg.utils.multiprocessing_utils import (
    send_list_to_workers,
    send_sublists_to_workers,
)


def worker(full_img_list, start_idx: int, end_idx: int, kwargs):
    """
    Worker process.
    """
    print(f"Processing {start_idx}->{end_idx}")

    # process each image between start_idx and end_idx
    for img_idx in range(start_idx, end_idx):
        img_fpath = full_img_list[img_idx]
        print(f'Converting {img_idx}"th RGB to grayscale object class img...')
        print(img_fpath)
        # img = imageio.imread(img_fpath)


def test_send_list_to_workers() -> None:
    """ """
    full_img_list = range(7)
    send_list_to_workers(
        num_processes=5, list_to_split=full_img_list, worker_func_ptr=worker
    )


def subset_worker(img_fpath, kwargs):
    """ """
    print("kwargs ", kwargs)
    print(img_fpath)


def test_send_sublists_to_workers() -> None:
    """ """
    full_img_list = range(7)
    send_sublists_to_workers(
        num_processes=4, list_to_split=full_img_list, worker_func_ptr=subset_worker
    )


class WorkerSpawner:
    def __init__(self):
        pass

    def subset_worker(self, img_fpath, kwargs):
        """ """
        print("kwargs ", kwargs)
        print(img_fpath)


def test_send_sublists_to_workers_method() -> None:
    """ """
    ws = WorkerSpawner()
    full_img_list = range(7)
    send_sublists_to_workers(
        num_processes=4, list_to_split=full_img_list, worker_func_ptr=ws.subset_worker
    )


test_send_sublists_to_workers_method()
