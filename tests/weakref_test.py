import weakref


class AnyObject:
    pass


def test_weakref_identity():
    any_object = AnyObject()
    ref = weakref.ref(any_object)
    assert ref() == any_object
    assert ref() is any_object
    assert ref is not any_object
    assert ref != any_object
