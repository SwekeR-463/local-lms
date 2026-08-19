#!/usr/bin/env python3
import runpy, signal, sys

signal.alarm(10)
f = runpy.run_path(sys.argv[1])["batch_iterable"]
assert f([], 3) == []
assert f(range(7), 3) == [[0, 1, 2], [3, 4, 5], [6]]
assert f((x * x for x in range(4)), 2) == [[0, 1], [4, 9]]
assert f("abc", 1) == [["a"], ["b"], ["c"]]
for size in (0, -1, 1.5, True):
    try: f([1], size)
    except ValueError: pass
    else: raise AssertionError(size)
print("ok")
