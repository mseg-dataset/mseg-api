#!/usr/bin/python3

from typing import Any, Mapping


def convert_dictionaries(source_to_middle_dict, middle_to_sink_dict):
    """ """
    source_to_sink_dict = {}
    for source_key, middle_val in source_to_middle_dict.items():

        sink_val = middle_to_sink_dict[middle_val]
        source_to_sink_dict[source_key] = sink_val

    return source_to_sink_dict


def reverse_dict(forward_dict: Mapping[Any, Any]):
    """
    We ensure that a 1:1 mapping is even possible (unique keys and values).

    The following would fail:
            {'a': 1, 'b': 2, 'c': 3, 'd': 3}
    """
    keys = list(forward_dict.keys())
    values = list(forward_dict.values())
    assert len(set(keys)) == len(keys)
    assert len(set(values)) == len(values)
    backward_dict = {v: k for k, v in forward_dict.items()}
    return backward_dict


def dict_is_equal(dict1, dict2):
    """ """
    assert sorted(list(dict1.keys())) == sorted(list(dict2.keys()))
    assert sorted(list(dict1.values())) == sorted(list(dict2.values()))

    for k1, v1 in dict1.items():
        assert v1 == dict2[k1]


def test_reverse_dict1():
    """ """
    forward_dict = {"a": 1, "b": 2, "c": 3}
    backward_dict = reverse_dict(forward_dict)
    backward_dict_gt = {1: "a", 2: "b", 3: "c"}
    dict_is_equal(backward_dict, backward_dict_gt)


if __name__ == "__main__":

    test_reverse_dict1()
