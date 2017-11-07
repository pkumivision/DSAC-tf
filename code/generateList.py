import os
from os import path

train_set = [1,2,4,6]
test_set = [3, 5]

root_dir = r'/scratch/chil/7scenes/chess'
out_dir = r'../list'

def tolist(outpath, sets):
    out = open(outpath, 'w')
    for no in sets:
        seq_name = 'seq-{:0>2d}'.format(no)
        seq_dir = path.join(root_dir, seq_name)
        l = len(os.listdir(seq_dir))/3
        for i in xrange(l):
            frame_name = seq_name+'/frame-{:0>6d}'.format(i)
            print >> out, frame_name+'.color.png', seq_name+'/{:0>6d}.npy'.format(i) 

tolist(path.join(out_dir,'train.list'), train_set)
tolist(path.join(out_dir,'test.list'), test_set)
