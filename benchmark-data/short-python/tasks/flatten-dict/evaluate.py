#!/usr/bin/env python3
import runpy, signal, sys

signal.alarm(10)
f = runpy.run_path(sys.argv[1])["flatten_dict"]
assert f({}) == {}
source = {"a": {"b": 1, "c": {}}, "d": [2, 3], "e": 0}
assert f(source) == {"a.b": 1, "a.c": {}, "d": [2, 3], "e": 0}
assert source == {"a": {"b": 1, "c": {}}, "d": [2, 3], "e": 0}
assert f({"a": {"b": {"c": None}}}, "/") == {"a/b/c": None}
assert f({"x": False}) == {"x": False}
print("ok")
