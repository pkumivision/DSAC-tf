import os
from os import path

train_set = [1,2,4,6]
test_set = [3, 5]

root_dir = r'/scratch/chil/7scenes/chess'
out_dir = r'../list'

def tolist_obj(outpath, sets):
    out = open(outpath, 'w')
    for no in sets:
        seq_name = 'seq-{:0>2d}'.format(no)
        seq_dir = path.join(root_dir, seq_name)
        l = len(os.listdir(seq_dir))/4
        for i in xrange(l):
            frame_name = seq_name+'/frame-{:0>6d}'.format(i)
            print >> out, frame_name+'.color.png', seq_name+'/{:0>6d}.npy'.format(i)

def tolist_score(outpath, sets):
    out = open(outpath, 'w')
    for no in sets:
        seq_name = 'seq-{:0>2d}'.format(no)
        seq_dir = path.join(root_dir, seq_name)
        l = len(os.listdir(seq_dir))/4
        for i in xrange(l):
            frame_name = seq_name+'/frame-{:0>6d}'.format(i)
            print >> out, frame_name+'.color.png', frame_name+'.pose.txt'.format(i)

tolist_obj(path.join(out_dir,'train_obj.list'), train_set)
tolist_obj(path.join(out_dir,'test_obj.list'), test_set)
tolist_score(path.join(out_dir,'train_score.list'), train_set)
tolist_score(path.join(out_dir,'test_score.list'), test_set)
