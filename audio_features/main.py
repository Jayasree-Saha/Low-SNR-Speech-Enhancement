import argparse
import torch 
import numpy as np

import random
from audio import *
from glob import glob
from hparams import hparams

parser = argparse.ArgumentParser(description='Audio features')

parser.add_argument('--root', default='/home/codes/2023',type=str, help='root of the data')


args = parser.parse_args()


all_wavs=glob(args.root+"/*.wav")
#pass through all data

for wav_f in all_wavs:
	wav=load_wav(wav_f, hparams.sample_rate)
	mels=melspectrogram(wav,hparams)
	#testing
	re_wav=inv_mel_spectrogram(mels, hparams)
	save_wav(re_wav, hparams.sample_rate, wav_f.replace(".wav","_inv.wav"))
