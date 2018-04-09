from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import nupic

from nupic.encoders import RandomDistributedScalarEncoder
from nupic.encoders.date import DateEncoder
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory

from scipy.stats import norm


class Encoder:
    def __init__(self, variable, encoders):
        self.variable = variable
        self.encoders = encoders


def multiencode(encoders, Data, iter):
    res = [0]
    for x in encoders:
        for y in x.encoders:
            if x.variable != '_index':
                exec("enc = " + y + ".encode(Data['" + x.variable + "'][iter])")
            else:
                exec("enc = " + y + ".encode(Data.index[iter])")
            res = np.concatenate([res, enc])
    return res[1:]