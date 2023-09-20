from audio_features.audio import *
import torch
import numpy as np
def convert_spec_to_wav(linear_spec_batch, hparams):
	batch, _, _=linear_spec_batch.shape
	spec=linear_spec_batch[0, :, :].T
	wav=inv_linear_spectrogram(spec.numpy(),hparams)
	'''
	for b in range(1, batch):
		wav=inv_linear_spectrogram(linear_spec_batch[b, :, :].T.numpy(),hparams)
		wav_batch=np.stack((wav_batch,wav), axis=0)
	print("wav_batch:",wav_batch.shape)
	'''
	return wav


def convert_mel_to_wav(mel_spec_batch, hparams):
	batch, _, _=mel_spec_batch.shape
	mels_batch=inv_mel_spectrogram(mel_spec_batch[0, :].T.numpy(), hparams)
	for b in range(1, batch):
		mels=inv_mel_spectrogram(mel_spec_batch[b, :].T.numpy(), hparams)
		mels_batch=np.concatenate((mels_batch, mels), axis=0)
	return mels_batch