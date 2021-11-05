#!/usr/bin/python3


def dict_is_equal(dict1, dict2):
    """ """
    assert set(dict1.keys()) == set(dict2.keys())
    assert set(dict1.values()) == set(dict2.values())

    for k1, v1 in dict1.items():
        assert v1 == dict2[k1]
