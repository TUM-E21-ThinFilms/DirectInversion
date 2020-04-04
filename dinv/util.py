import numpy
import time

global _bm_before, _bm_after, _debug

_bm_before = 0
_bm_after = 0

"""
Some utility function for benchmarking. Only active when _debug is set to True.
"""
try:
    if _debug is False:
        _debug = False
except:
    _debug = False


def benchmark_start():
    global _bm_after, _bm_before
    _bm_before = time.time()


def benchmark_stop(text):
    global _bm_after, _bm_before
    _bm_after = time.time()
    diff = _bm_after - _bm_before
    if _debug:
        print(text.format(str(diff) + " s"))

def _vslice(array, selector, dstart=0, dstop=0):
    start = array[0]
    stop = array[-1]
    steps = 1

    if len(selector) == 1:
        stop = selector[0]
    elif len(selector) > 1:
        start, stop = selector[0:2]
        if len(selector) == 3:
            steps = selector[2]
    if start is not None:
        ind_start = numpy.argmin(array <= start) + dstart
    else:
        ind_start = 0

    if stop is not None:
        ind_stop = numpy.argmax(array > stop) + dstop
        if ind_stop == 0:
            ind_stop = None
    else:
        ind_stop = None

    if steps is None:
        steps = 1

    return slice(ind_start, ind_stop, steps)


def islice(iterable, *selectors):
    """
    Slices a iterable using selectors and concatenates the result

    :param iterable:
    :param selectors:
    :return:
    """
    return numpy.concatenate([iterable[slice(*s)] for s in selectors])

def vslice(iterable, *selectors, dstart=0, dstop=0):
    """
    Slices an iterable using the values of selectors and concatenates the result

    This is useful if you have an array and you want to slice based on the values. Usually the slicing is
    based on the index of the array, here it is based on the value.

    :Example:
    If you want to have all values between 0.2 and 0.3, and 0.8 and 0.9 then you can use this
    >>> a = numpy.linspace(0, 1, 200)
    >>> b = vslice(a, (0.2, 0.3), (0.8, 0.9))


    :param iterable:
    :param selectors:
    :param dstart: delta index for start
    :param dstop: delta index for stop
    :return:
    """
    return numpy.concatenate([iterable[_vslice(iterable, s, dstart=dstart, dstop=dstop)] for s in selectors])
