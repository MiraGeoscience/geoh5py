import weakref

import pytest

from geoh5io.shared import weakref_utils


class AnyObject:
    pass


def test_remove_none_referents():
    some_dict = dict()
    some_dict["gone"] = weakref.ref(AnyObject())
    bound_object = AnyObject()
    some_dict["there"] = weakref.ref(bound_object)

    assert "gone" in some_dict
    assert some_dict["gone"]() is None
    assert some_dict["there"]() is bound_object
    weakref_utils.remove_none_referents(some_dict)
    assert "gone" not in some_dict


def test_get_clean_ref():
    some_dict = dict()
    some_dict["gone"] = weakref.ref(AnyObject())
    bound_object = AnyObject()
    some_dict["there"] = weakref.ref(bound_object)

    assert "gone" in some_dict
    assert some_dict["gone"]() is None
    assert some_dict["there"]() is bound_object
    assert weakref_utils.get_clean_ref(some_dict, "there") is bound_object
    assert weakref_utils.get_clean_ref(some_dict, "gone") is None
    assert "gone" not in some_dict
    assert "there" in some_dict
    assert some_dict["there"]() is bound_object


def test_insert_once():
    some_dict = dict()
    some_dict["gone"] = weakref.ref(AnyObject())
    bound_object = AnyObject()
    some_dict["there"] = weakref.ref(bound_object)

    assert "gone" in some_dict
    assert some_dict["gone"]() is None
    assert some_dict["there"]() is bound_object

    other = AnyObject()
    with pytest.raises(RuntimeError) as error:
        weakref_utils.insert_once(some_dict, "there", other)
    assert "Key 'there' already used" in str(error.value)

    weakref_utils.insert_once(some_dict, "gone", other)
    assert some_dict["gone"]() is other
