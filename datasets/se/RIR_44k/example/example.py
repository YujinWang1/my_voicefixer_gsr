#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rir.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/24/21 2:31 AM   Haohe Liu      1.0         None
'''


import numpy as np
import scipy.signal as s
import librosa
import soundfile as sf

def reverb_rir(frames, rir):
    orig_frames_shape = frames.shape
    frames, filter = np.squeeze(frames), np.squeeze(rir)
    frames = s.convolve(frames, filter)
    actlev = np.max(np.abs(frames))
    if (actlev > 0.99):
        frames = (frames / actlev) * 0.98
    frames = frames[:orig_frames_shape[0]]
    return frames

if __name__ == '__main__':
    frames,_ = librosa.load("example.wav",sr=44100)
    rir = np.load("rir_cardioid_rt60_0.28_room_8.4_11.08_11.71_mic_2.9_9.48_0.54_source_2.73_10.16_8.81.npy")
    frames = reverb_rir(frames,rir=rir)
    sf.write(file="with_reverb.wav",data=frames,samplerate=44100)

