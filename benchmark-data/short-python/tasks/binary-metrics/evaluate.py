#!/usr/bin/env python3
import runpy, signal, sys

signal.alarm(10)
f = runpy.run_path(sys.argv[1])["binary_metrics"]
m = f(6, 8, 2, 4)
assert m == {"accuracy": 0.7, "precision": 0.75, "recall": 0.6, "f1": 2 * 0.75 * 0.6 / 1.35, "false_positive_rate": 0.2}
assert f(0, 5, 2, 3)["f1"] == 0.0
assert f(0, 0, 0, 0) == dict.fromkeys(("accuracy", "precision", "recall", "f1", "false_positive_rate"), 0.0)
for args in [(-1, 0, 0, 0), (1.0, 0, 0, 0), (True, 0, 0, 0)]:
    try: f(*args)
    except ValueError: pass
    else: raise AssertionError(args)
print("ok")
