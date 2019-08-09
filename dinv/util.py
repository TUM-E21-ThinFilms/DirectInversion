import time

global _bm_before, _bm_after, _debug

_bm_before = 0
_bm_after = 0

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