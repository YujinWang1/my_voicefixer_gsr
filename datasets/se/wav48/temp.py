#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   convert_wav_flac.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/14/21 8:23 PM   Haohe Liu      1.0         None
'''

 


import torch
import wave
import json
import os
import glob
from progressbar import *

EPS=1e-8


import torch
import time
from pynvml import *

def convert_wav_to_flac(dir):
    current = "wav"
    files = glob.glob(os.path.join(dir, "*." + current)) + \
                glob.glob(os.path.join(dir, "*/*." + current)) + \
                glob.glob(os.path.join(dir, "*/*/*." + current)) + \
                glob.glob(os.path.join(dir, "*/*/*/*." + current)) + glob.glob(os.path.join(dir, "*/*/*/*/*." + current))
    widgets = [
        "Convert wav to flac",
        ' [', Timer(), '] ',
        Bar(),
        ' (', ETA(), ') ',
    ]
    pbar = ProgressBar(widgets=widgets).start()
    for i,path in enumerate(files):
        if (current == "wav"):
            cmd = "sox " + path + " " + path[:-4] + ".flac"
        if (current == "flac"):
            cmd = "sox " + path + " " + path[:-5] + ".flac"
        os.system(cmd)
        os.remove(path)
        pbar.update(int((i / (len(files) - 1)) * 100))
    pbar.finish()


convert_wav_to_flac(".")
