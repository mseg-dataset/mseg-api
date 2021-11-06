#!/usr/bin/python3

import json
import os
from typing import Any, Dict, Union


def read_json_file(fpath: str):
    """
    Args:
        fpath: string, representing file path
    """
    with open(fpath, "rb") as f:
        json_data = json.load(f)
    return json_data


def save_json_dict(json_fpath: Union[str, "os.PathLike[str]"], dictionary: Dict[Any, Any]) -> None:
    """Save a Python dictionary to a JSON file.
    
    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)
