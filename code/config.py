import os
import os.path as osp
import numpy as np
from distutils import spawn
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.CNN_OBJ_MAXINPUT = 100.0
__C.EPS = 0.00000001
__C.REFTIMES = 8
__C.INLIERTHRESHOLD2D = 10
__C.INLIERCOUNT = 20
__C.SKIP = 10
__C.HYPNUM = 4