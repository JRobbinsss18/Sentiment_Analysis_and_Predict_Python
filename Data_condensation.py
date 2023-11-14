import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings

# TEST CODE JUST TO MAKE SURE CODE COMPILES
def maximum(a, b):
     
    if a >= b:
        return a
    else:
        return b
a = 2
b = 4
print(maximum(a, b)) 
