import random
import networkx as nx
import numpy as np
import copy
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import os, sys

# Reload past data to regenerate graphs
def read_csv_custom(filename):
     res = pd.read_csv(filename,header=0,index_col=0)
     res.columns = res.columns.astype(float)
     return res

# This rogram merges data collected from 32 experiments that stored the results in the directories indicated in 'dirs'
simulations = ["ring_10","petersen","random_10","complete_8"]
dirs = ["r01","r02","r03","r04","r05","r06","r07","r08","r09","r10","r11","r12","r13","r14","r15","r16","r17","r18","r19","r20","r21","r22","r23","r24","r25","r26","r27","r28","r29","r30","r31","r32"]
merge_dir = Path("merge_dir")

for s in simulations:
    l = list((Path(dirs[0])/s).glob('*.csv'))
    for f in l:
        print(f"Loading {f}")
        df = read_csv_custom(f)
        for d in dirs[1:]:
            p = Path(d)/s/f.name
            if p.is_file():
                print(f"Merging {f} with {p}")
                df_new = read_csv_custom(p)
                df = pd.concat([df,df_new])
            else:
                print(f"Could not find file {p}")
        # save merged file in last directory
        (merge_dir/s).mkdir(parents=True, exist_ok=True)
        df.to_csv(merge_dir/s/f.name)
        