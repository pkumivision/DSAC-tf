import os
from os import path

src_dir = r'/scratch/chil/7scenes_align/chess'
tgt_dir = r'../image'

subdirs = ['seq-01','seq-02','seq-03','seq-04','seq-05','seq-06']

for subdir in subdirs:
    seq_dir = path.join(src_dir, subdir)
    for i in xrange(1000):
        npyname = '{:0>6d}.npy'.format(i)
        src_path = path.join(seq_dir, npyname)
        tgt_path = path.join(tgt_dir, subdir, npyname)
        os.system('ln -s {} {}'.format(src_path, tgt_path))