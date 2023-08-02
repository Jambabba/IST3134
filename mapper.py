#!/usr/bin/env python3

import sys

for line in sys.stdin:
        # remove whitespaces
        line = line.strip()
        # split line into words
        words = line.split(',')
        # map counter
        for word in words:
                print(f"{word}\t1")

