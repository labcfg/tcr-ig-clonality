from tqdm import tqdm, trange, tqdm_notebook
import pandas as pd
import os.path
import sys, os, re
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import Counter
from operator import itemgetter
from sklearn.externals.joblib import Parallel, delayed
#from Levenshtein import distance


deg_nucl = {'A':['A'],
            'G':['G'],
            'T':['T'],
            'C':['C'],
            'N':['A','C','G','T']}

def hamming(x1, x2, m):
    if len(x1) != len(x2):
        return(False)
    j = 0
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            j += 1
            if j > m:
                return (False)
    return (True)

"""
def levenshtein(s1, s2, m):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, m)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 not in deg_nucl[c2])
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    if previous_row[-1] > m:
        return (False)
    else:
        return (True)
"""

def count(inputdir, outputdir, filename, mist, top):
    df = pd.read_table(inputdir + filename)
    readsname = filename.split('_reads')[0]
    readsname = readsname.rsplit('.', 1)[0]
    df_group = df.groupby(['FAMILY'])
    colnames = list(df.columns)
    colnames.insert(0, 'AMOUNT')
    colnames.append('VARIOUS')
    colnames.insert(2, 'PERCENT')
    df_out = pd.DataFrame(columns=colnames)
    if len(list(df.READ)) == 0:
        df_out.to_csv(outputdir + readsname + '_clones.txt', sep='\t', index=False)
    else:
        for name, group in tqdm_notebook(df_group):
            reads = dict(Counter(list(group.READ)))
            reads_revsort = sorted(reads.items(), key=itemgetter(1))
            reads_revsort.insert(0, ('RRRRRRRRRRRRRRRRR', 1))
            reads_revsort.insert(0, ('FFFFFFFFFFFFFFFFF', 1))
            clones_seq = []
            clones_amount = []
            clones_various = []
            while (len(reads_revsort) > 2):
                #print(len(reads_revsort))
                current_read = reads_revsort[-1][0]
                current_amount = reads_revsort[-1][1]
                current_various = 1
                elem4remove = []
                del reads_revsort[-1]
                for idx, r_a in enumerate(reads_revsort):
                    read = r_a[0]
                    amount = r_a[1]
                    if hamming(current_read, read, mist):
                        current_amount += amount
                        current_various += 1
                        elem4remove.append(idx)
                offset = 0
                elem4remove = sorted(elem4remove)
                for idx in elem4remove:
                    del reads_revsort[idx-offset]
                    offset += 1
                clones_seq.append(current_read)
                clones_amount.append(current_amount)
                clones_various.append(current_various)
            clones_persent = [x/sum(clones_amount) for x in clones_amount]
            clones = sorted(zip(clones_seq, clones_amount, clones_various, clones_persent),
                            key=itemgetter(1), reverse=True)
            if top == 'ALL':
                continue
            else:
                clones = clones[:top]
            for i in range(len(clones)):
                needed_row = group.loc[group.READ == clones[i][0]].iloc[0]
                needed_row['AMOUNT'] = clones[i][1]
                needed_row['VARIOUS'] = clones[i][2]
                needed_row['PERCENT'] = clones[i][3]
                df_out = df_out.append(needed_row)
        df_out.sort_values(['AMOUNT'], ascending = False, inplace = True)
        df_out.to_csv(outputdir + readsname + '_clones.txt', sep='\t', index=False)


def parallel_files(inputdir, outputdir, filename, mist, top):
    file, ext = os.path.splitext(filename)
    if ext == '.txt' and re.search('_reads', filename):
        count(inputdir, outputdir, filename, mist, top)


def main(inputdir, outputdir, mist, top, n_core):
    inputdir = os.path.abspath(inputdir) + '/'
    outputdir = os.path.abspath(outputdir) + '/'

    # Read files in folder
    onlyfiles = [f for f in listdir(inputdir) if isfile(join(inputdir, f))]

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    Parallel(n_jobs=n_core)(delayed(parallel_files)(inputdir,
                                                    outputdir,
                                                    filename,
                                                    mist,
                                                    top) for filename in onlyfiles)
