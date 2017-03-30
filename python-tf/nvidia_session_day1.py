import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
logdir = './'

import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from IPython.display import clear_output, Image, display, HTML