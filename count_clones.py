from Bio import SeqIO
from Bio.Seq import Seq
import sys, os, re
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import namedtuple
import pandas as pd
from tqdm import tqdm_notebook, tnrange
from collections import Counter, defaultdict
from operator import itemgetter
from sklearn.externals.joblib import Parallel, delayed

find_p = namedtuple('find_p', 'is_primer_found read family gene_fwd name_fwd name_rev primer_fwd primer_rev ham shift')

deg_nucleotide = {'A': ['A'],
                  'G': ['G'],
                  'T': ['T'],
                  'C': ['C'],
                  'B': ['C', 'G', 'T'],
                  'D': ['A', 'G', 'T'],
                  'H': ['A', 'C', 'T'],
                  'K': ['G', 'T'],
                  'M': ['A', 'C'],
                  'N': ['A', 'C', 'G', 'T'],
                  'R': ['G', 'A'],
                  'S': ['G', 'C'],
                  'V': ['A', 'C', 'G'],
                  'W': ['A', 'T'],
                  'Y': ['C', 'T'],
                  'X': ['A', 'C', 'G', 'T']}

deg_nucleotide_keys = list(deg_nucleotide.keys())


def hamming(x1, x2, m):
    j = 0
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            j += 1
            if j > m:
                return False, None
    return True, j


def hamming_mod(x1, x2, m):
    j = 0
    for i in range(len(x1)):
        if not x2[i] in deg_nucleotide[x1[i]]:
            j += 1
            if j > m:
                return False, None
    return True, j


def count_lines(filepath):
    with open(filepath) as f:
        return sum(1 for _ in f)


def find_primer(raw_r1, raw_r2, primers, mist, shift, lb, seq_len, is_direct_ordered):
    if is_direct_ordered:
        genes_for_r1 = set(['V', 'v', 'D', 'd'])
        genes_for_r2 = set(['J', 'j'])
    else:
        genes_for_r1 = set(['J', 'j'])
        genes_for_r2 = set(['V', 'v', 'D', 'd'])
    for index1, row1 in primers[primers.gene.isin(genes_for_r1)].iterrows():
        for i in range(shift + 1):
            if re.search('[' + ''.join(deg_nucleotide_keys) + ']', row1['seq'].upper()[lb:]):
                ham = hamming_mod(row1['seq'].upper()[lb:],
                                  str(raw_r1.seq)[i:len(row1['seq'][lb:]) + i], mist)
                if ham[0]:
                    best_primer_r1 = (ham[1],
                                      row1['family'],
                                      row1['gene'],
                                      row1['name'],
                                      str(raw_r1.seq)[i:len(row1['seq'][lb:]) + i])
            else:
                ham = hamming(row1['seq'].upper()[lb:],
                              str(raw_r1.seq)[i:len(row1['seq'][lb:]) + i], mist)
                if ham[0]:
                    best_primer_r1 = (ham[1],
                                      row1['family'],
                                      row1['gene'],
                                      row1['name'],
                                      str(raw_r1.seq)[i:len(row1['seq'][lb:]) + i])
            if ham[0]:
                for index2, row2 in primers[(primers.gene.isin(genes_for_r2)) & (primers.family == best_primer_r1[1])].iterrows():
                    for j in range(shift + 1):
                        if re.search('[' + ''.join(deg_nucleotide_keys) + ']', row2['seq'].upper()[lb:]):
                            ham = hamming_mod(row2['seq'].upper()[lb:],
                                              str(raw_r2.seq)[j:len(row2['seq'][lb:]) + j], mist)
                            if ham[0]:
                                best_primer_r2 = (ham[1],
                                                  row2['name'],
                                                  str(raw_r2.seq)[j:len(row2['seq'][lb:]) + j])
                        else:
                            ham = hamming(row2['seq'].upper()[lb:],
                                          str(raw_r2.seq)[j:len(row2['seq'][lb:]) + j], mist)
                            if ham[0]:
                                best_primer_r2 = (ham[1],
                                                  row2['name'],
                                                  str(raw_r2.seq)[j:len(row2['seq'][lb:]) + j])
                        if ham[0]:
                            return (find_p(is_primer_found=True,
                                           read=str(raw_r2.seq)[len(row2['seq'][lb:]) + j:seq_len],
                                           family=best_primer_r1[1],
                                           gene_fwd=best_primer_r1[2],
                                           name_fwd=best_primer_r1[3],
                                           name_rev=best_primer_r2[1],
                                           primer_fwd=best_primer_r1[4],
                                           primer_rev=best_primer_r2[2],
                                           ham=max([best_primer_r1[0], best_primer_r2[0]]),
                                           shift=max([i, j])))
                return (find_p(is_primer_found=False,
                               read=None,
                               family=None,
                               gene_fwd=None,
                               name_fwd=None,
                               name_rev=None,
                               primer_fwd=None,
                               primer_rev=None,
                               ham=None,
                               shift=None))
    return (find_p(is_primer_found=False,
                   read=None,
                   family=None,
                   gene_fwd=None,
                   name_fwd=None,
                   name_rev=None,
                   primer_fwd=None,
                   primer_rev=None,
                   ham=None,
                   shift=None))


def calc_reads(inputdir, filename1, filename2, outputdir, primers, mist, shift, lookback, family_stats, seq_len, is_direct_ordered):
    original_R1_reads = SeqIO.parse(inputdir + filename1, "fastq")
    original_R2_reads = SeqIO.parse(inputdir + filename2, "fastq")
    readsname = filename1.split('R1')[0]
    readsname = readsname.rsplit('.', 1)[0]
    f_out_reads = open(outputdir + readsname + '_reads.txt', 'w')
    f_out_reads.write('FAMILY\tREAD\tGENE_FWD\tGENE_REV\tNAME_FWD\tNAME_REV\t' +
                      'PRIMER_FWD\tPRIMER_REV\n')
    f_out_errors_r1 = open(outputdir + readsname + '_R1_errors.fastq', 'w')
    f_out_errors_r2 = open(outputdir + readsname + '_R2_errors.fastq', 'w')
    error_stats = [0, 0]  # 0 - all, 1 - bad
    hamming_stats = []
    shift_stats = []
    lookback_stats = []

    bar = tnrange(int(count_lines(inputdir + filename1) / 4), desc=readsname)
    original_R12 = zip(original_R1_reads, original_R2_reads)

    for i in bar:
        r1, r2 = next(original_R12)
        stat = find_primer(r1, r2, primers, mist, shift, 0, seq_len, is_direct_ordered)
        k = 0
        if not stat.is_primer_found:
            for k in range(1, lookback + 1):
                stat = find_primer(r1, r2, primers, mist, 0, k, seq_len, is_direct_ordered)
                if stat.is_primer_found:
                    break
        error_stats[0] += 1
        if stat.is_primer_found:
            f_out_reads.write(stat.family + '\t' +
                              stat.read + '\t' +
                              stat.gene_fwd + '\t' +
                              'J' + '\t' +
                              stat.name_fwd + '\t' +
                              stat.name_rev + '\t' +
                              stat.primer_fwd + '\t' +
                              stat.primer_rev + '\n')
            family_stats[stat.family] += 1
            hamming_stats.append(stat.ham)
            shift_stats.append(stat.shift)
            lookback_stats.append(k)
        else:
            error_stats[1] += 1
            f_out_errors_r1.write(r1.format('fastq'))
            f_out_errors_r2.write(r2.format('fastq'))
    f_out_reads.close()
    f_out_errors_r1.close()
    f_out_errors_r2.close()
    hamming_stats = sorted(dict(Counter(hamming_stats)).items(), key=itemgetter(0))
    shift_stats = sorted(dict(Counter(shift_stats)).items(), key=itemgetter(0))
    lookback_stats = sorted(dict(Counter(lookback_stats)).items(), key=itemgetter(0))
    stats = dict()
    stats['error'] = error_stats
    stats['family'] = family_stats
    stats['ham'] = ','.join([str(h) + ':' + str(a) for h, a in hamming_stats])
    stats['shift'] = ','.join([str(s) + ':' + str(a) for s, a in shift_stats])
    stats['lb'] = ','.join([str(l) + ':' + str(a) for l, a in lookback_stats])
    bad = float('{:0.2f}'.format(stats['error'][1] / stats['error'][0]))
    good = float('{:0.2f}'.format(1 - bad))
    family_stats = sorted(stats['family'].items(), key=itemgetter(0))
    stats_line = '\t'.join([readsname, str(stats['error'][0]), str(bad), str(good),
                            stats['ham'], stats['shift'], stats['lb'], '\t'.join(str(i) for f, i in family_stats)])

    return stats_line


def parallel_files(filename1, filename2, inputdir, outputdir, primers,
                   mist, shift, lookback, seq_len, is_direct_ordered):
    ext1, inputfile1 = os.path.splitext(filename1)
    ext2, inputfile2 = os.path.splitext(filename2)
    readsname = filename1.split('R1')[0]
    readsname = readsname.rsplit('.', 1)[0]
    family_stats = dict()
    for f in list(primers.family):
        family_stats[f] = 0
    stats = calc_reads(inputdir, filename1, filename2, outputdir, primers,
                       mist, shift, lookback, family_stats, seq_len, is_direct_ordered)
    return stats


def main(inputdir, outputdir, primersfile, statfile, mist, shift, lookback, seq_len, is_direct_ordered, n_core):
    inputdir = os.path.abspath(inputdir) + '/'
    outputdir = os.path.abspath(outputdir) + '/'

    # Read files in folder
    onlyfiles = [f for f in listdir(inputdir) if isfile(join(inputdir, f))]
    r1_files = {}
    r2_files = {}
    for filename in onlyfiles:
        filename = filename.rstrip()
        if re.search('R1', filename):
            key_filename = filename.split('R1')[0]
            r1_files[key_filename] = filename
        elif re.search('R2', filename):
            key_filename = filename.split('R2')[0]
            r2_files[key_filename] = filename

    conform_files = []
    nonconform_files = []

    for key in r1_files:
        if key in r2_files:
            conform_files.append([r1_files[key], r2_files[key]])
            del r2_files[key]
        else:
            nonconform_files.append(r1_files[key])

    nonconform_files = nonconform_files + list(r2_files.values())

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    primers = pd.read_csv(primersfile)
    statistics = open(outputdir + statfile, 'w')
    statistics.write('#parametrs: mist_for_primer=' + str(mist) +
                     ', shift_for_primer=' + str(shift) + '\n')
    statistics.write('READNAME\tREADS\tBAD\tGOOD\tHAMMING\tSHIFT\tLOOKBACK\t' +
                     '\t'.join(sorted(list(set(list(primers.family))))) + '\n')

    stats = Parallel(n_jobs=n_core)(delayed(parallel_files)(f1,
                                                            f2,
                                                            inputdir,
                                                            outputdir,
                                                            primers,
                                                            mist,
                                                            shift,
                                                            lookback,
                                                            seq_len,
                                                            is_direct_ordered) for f1, f2 in conform_files)

    for line in stats:
        statistics.write(line + '\n')
