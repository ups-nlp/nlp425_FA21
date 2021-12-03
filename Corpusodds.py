# These are some imports that might be useful.
from collections import Counter
import math
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import os
import re
import statistics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from string import punctuation
import re
words = []
f = open("movie_lines.txt", "r", encoding="ISO-8859-1")
lines = f.readlines()
for i in range(0, len(lines)):
    line = lines[i].rsplit(" +++$+++ ", 1)
    line2 = line[1].lower()
    line2 = re.sub(r"[,:.\"\-_;@#?!&$]+", '', line2)
    line2 = re.sub('\s+',' ', line2)
    words.append(line2)

myDict = {}
ntok= 0
ntyp = 0
for w in range(0, len(words)):
    for word in words[w].split(' '):
        if not word == '':
            if word in myDict:
                myDict[word] += 1
                ntok+= 1
            else:
                myDict[word] = 1
                ntyp+= 1

from collections import Counter
n=2
myCount = Counter()
def find_ngrams(input_list, n):
    phra= input_list.split()
    return zip(*[phra[v:] for v in range(n)])


for s in range(0, len(words)):
    ngrams = find_ngrams(words[s], n)
    myCount.update(ngrams)

myCount = {key:val for key, val in myCount.items() if val != 1}
    
myCount

# Returns a list of (word, count) tuples
# sorted_count = myCount.most_common(20)
# sorted_count
# print(myCount)
# f = open("move_2odds.txt", "w")
# f.writelines(myCount)

with open("move_2odds.txt", 'w') as f: 
    for key, value in myCount.items(): 
        f.write('%s:%s\n' % (key, value))