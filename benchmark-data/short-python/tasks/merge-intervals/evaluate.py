#!/usr/bin/env python3
import runpy, signal, sys

signal.alarm(10)
f = runpy.run_path(sys.argv[1])["merge_intervals"]
assert f([]) == []
assert f([(5, 7), (1, 4), (3, 6)]) == [(1, 7)]
assert f([(1, 2), (2, 3), (8, 9)]) == [(1, 3), (8, 9)]
assert f([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]
source = [(9, 10), (1, 2)]
assert f(source) == [(1, 2), (9, 10)] and source == [(9, 10), (1, 2)]
assert f((x for x in [(0, 0), (2, 2)])) == [(0, 0), (2, 2)]
print("ok")
