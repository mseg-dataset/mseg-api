#!/usr/bin/python3

import math
import multiprocessing
import pdb
from tqdm import tqdm

from typing import Any, Callable, List


def send_list_to_workers(num_processes: int, list_to_split: List[Any], worker_func_ptr: Callable, **kwargs) -> None:
    """Given a list of work, and a desired number of n workers, launch n worker processes
    that will each process 1/nth of the total work.

    Args:
        num_processes: integer, number of workers to launch
        list_to_split:
        worker_func_ptr: function pointer
        **kwargs:
    """
    jobs = []
    num_items = len(list_to_split)
    print(f"Will split {num_items} items between {num_processes} workers")
    chunk_sz = math.ceil(num_items / num_processes)
    for i in range(num_processes):

        start_idx = chunk_sz * i
        end_idx = start_idx + chunk_sz
        end_idx = min(end_idx, num_items)
        # print(f'{start_idx}->{end_idx}')

        p = multiprocessing.Process(target=worker_func_ptr, args=(list_to_split, start_idx, end_idx, kwargs))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()


def send_sublists_to_workers(num_processes: int, list_to_split: List[Any], worker_func_ptr, **kwargs) -> None:
    """Given a list of work, and a desired number of n workers, launch n worker processes
    that will each process 1/nth of the total work.

    Args:
        num_processes: integer, number of workers to launch
        list_to_split:
        worker_func_ptr: function pointer
        **kwargs:
    """
    jobs = []
    num_items = len(list_to_split)
    print(f"Will split {num_items} items between {num_processes} workers")
    chunk_sz = math.ceil(num_items / num_processes)

    def execute_on_list_subset(worker_func_ptr: Callable, list_to_split, start_index: int, end_index: int, kwargs) -> None:
        """
        Each process will be calling the worker_func_ptr over and over again.
        Helps with readability, but lots of overhead from the call stack.

        Args:
            worker_func_ptr:
            list_to_split:
            start_index:
            end_index:
            kwargs:
        """
        print(f"start_index {start_index}, end_index {end_index}")
        for i in range(start_index, end_index):
            worker_func_ptr(list_to_split[i], kwargs)

    for i in range(num_processes):

        start_idx = chunk_sz * i
        end_idx = start_idx + chunk_sz
        end_idx = min(end_idx, num_items)

        p = multiprocessing.Process(
            target=execute_on_list_subset, args=(worker_func_ptr, list_to_split, start_idx, end_idx, kwargs)
        )
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()


def send_list_to_workers_new(num_processes: int, list_to_split: List[Any], worker_func_ptr: Callable) -> None:
    """Given a list of work, and a desired number of n workers, launch n worker processes
    that will each process 1/nth of the total work.

    Args:
        num_processes: integer, number of workers to launch
        list_to_split:
        worker_func_ptr: function pointer
        **kwargs:
    """
    jobs = []
    num_items = len(list_to_split)
    print(f"Will split {num_items} items between {num_processes} workers")
    chunk_sz = math.ceil(num_items / num_processes)

    def worker_func_ptr_list(worker_func_ptr, list_to_split, start_index, end_index):
        # worker_func_ptr()
        print(f"start_index {start_index}, end_index {end_index}")
        for i in tqdm(range(start_index, end_index)):
            worker_func_ptr(list_to_split[i])

    for i in range(num_processes):

        start_idx = chunk_sz * i
        end_idx = start_idx + chunk_sz
        end_idx = min(end_idx, num_items)
        # print(f'{start_idx}->{end_idx}')

        p = multiprocessing.Process(
            target=worker_func_ptr_list, args=(worker_func_ptr, list_to_split, start_idx, end_idx)
        )
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
