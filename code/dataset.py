import cv2
import numpy as np
import os
from os import path
import time

import properties as gp

class DataGenerator(object):
    def __init__(self, opt):
        self.opt = opt
        self.rgb_paths =[]
        self.obj_paths = []
        with open(opt.list) as f:
            for line in f.readlines():
                rgb, obj = line.strip().split(' ')
                self.rgb_paths.append(rgb)
                self.obj_paths.append(obj)
        self.indexs = np.arange(len(self.rgb_paths))
        self.total = len(self.rgb_paths) * self.opt.training_patches
        self._new_epoch()

    def _new_epoch(self):
        start_time = time.time()
        np.random.shuffle(self.indexs)
        self.objs = []
        self.patches = []
        for i in xrange(self.opt.training_images):
            curi = self.indexs[i]
            rgb = cv2.imread(path.join(self.opt.dataset_dir, self.rgb_paths[curi]))
            obj = np.load(path.join(self.opt.dataset_dir, self.obj_paths[curi]))
            rgb_h, rgb_w, rgb_c = rgb.shape
#            x, y = np.meshgrid(np.arange(rgb_w), np.arange(rgb_h))
#            valid_x, valid_y = np.where(obj[:,:,2]!=0)
#            valid_index = np.arange(len(valid_x))
#            np.random.shuffle(valid_index)

 #           valid_index = valid_index[:self.opt.training_patches]
 #           sx = x[valid_index]
 #           sy = y[valid_index]
 #           minx = sx - self.opt.input_size/2
 #           maxx = sx + self.opt.input_size/2
 #           miny = sy - self.opt.input_size/2
 #           maxy = sy + self.opt.input_size/2

 #           self.patches = rgb[valid_index]

            j = 0
            while j < self.opt.training_patches:
                x = np.random.randint(self.opt.input_size/2, rgb_w-self.opt.input_size/2)
                y = np.random.randint(self.opt.input_size/2, rgb_h-self.opt.input_size/2)
                if obj[y,x,2] == 0:
                    continue
                minx = x - self.opt.input_size/2
                maxx = x + self.opt.input_size/2
                miny = y - self.opt.input_size/2
                maxy = y + self.opt.input_size/2
                self.patches.append(rgb[miny:maxy,minx:maxx,:])
                self.objs.append(obj[y, x, :]/1000) #convert to mm
                j += 1

        assert len(self.patches) == self.opt.training_images*self.opt.training_patches
        self.step = 0
        if self.opt.time_info:
            print 'Generated {} patches from {} images in {}ms.'.format(len(self.patches),self.opt.training_images,(time.time()-start_time)*1000)

    def _next(self):
        if self.step == self.opt.training_images*self.opt.training_patches:
            self._new_epoch()
        patch = self.patches[self.step]
        obj = self.objs[self.step]
        self.step += 1
        return patch, obj

    def next_batch(self):
        data = []
        label = []
        for i in xrange(self.opt.batch_size):
            d, l = self._next()
            data.append(d)
            label.append(l)

        return data, label
