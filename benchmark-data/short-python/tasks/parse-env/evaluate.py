#!/usr/bin/env python3
import runpy, signal, sys

signal.alarm(10)
f = runpy.run_path(sys.argv[1])["parse_env"]
assert f("") == {}
assert f(" # comment\n A = one=two \nB=\" x \"\n\nA=last") == {"A": "last", "B": "\" x \""}
assert f("HASH=#literal\n X = y # literal") == {"HASH": "#literal", "X": "y # literal"}
for text, line in [("OK=1\nbad", "2"), (" =value", "1")]:
    try: f(text)
    except ValueError as e: assert line in str(e)
    else: raise AssertionError(text)
print("ok")
