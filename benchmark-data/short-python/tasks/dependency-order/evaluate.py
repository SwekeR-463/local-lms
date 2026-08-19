#!/usr/bin/env python3
import runpy, signal, sys

signal.alarm(10)
f = runpy.run_path(sys.argv[1])["dependency_order"]
assert f({}) == []
assert f({"deploy": ["test", "build"], "test": ["build"], "lint": []}) == ["build", "lint", "test", "deploy"]
assert f({"b": ["a"], "c": ["a"]}) == ["a", "b", "c"]
graph = {"b": ["a"]}
assert f(graph) == ["a", "b"] and graph == {"b": ["a"]}
for graph in ({"a": ["a"]}, {"a": ["b"], "b": ["a"]}):
    try: f(graph)
    except ValueError: pass
    else: raise AssertionError(graph)
print("ok")
