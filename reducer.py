#!/usr/bin/env/python3

from operator import itemgetter
import sys

current_word= None
current_count = 0
word = None

for line in sys.stdin:
        # remove whitespacees
        line = line.strip()
        # parse input from mapper.py
        word, count = line.split('\t', 1)
        # convert count to int
        try:
           count = int(count)
        except ValueError:
           continue 


        if current_word == word:
           current_count += count
        else: 
             if current_word:
                print(f"{current_word}\t{current_count}")
             current_count = count
             current_word = word

if current_word == word:
        print(f"{current_word}\t{current_count}")
