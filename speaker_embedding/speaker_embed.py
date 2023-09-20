import numpy as np
import os
import pickle
import argparse
from .audio.hparams import hparams as hp 
from tqdm import tqdm
import librosa
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd
import subprocess
import cv2
import torch.nn as nn


def _load(checkpoint_path):
	if torch.cuda.is_available():
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	# print("checkpoint:",checkpoint)
	return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, disc=False):

	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	# print (s.keys())
	# new_s = {}
	
	# for k, v in s.items():
	# 	# if n_gpu > 1:
	# 	if not k.startswith('module.'):
	# 		new_s['module.'+k] = v
	# 	else:
	# 		new_s[k] = v
	# 	# else:
	# 	# 	new_s[k.replace('module.', '')] = v
	model.load_state_dict(s)
			
	if not reset_optimizer:
		optimizer_state = checkpoint["optimizer"]
		if optimizer_state is not None:
			print("Load optimizer state from {}".format(path))
			optimizer.load_state_dict(optimizer_state)

	epoch_resume=0
	if disc==False:
		epoch_resume = checkpoint['epoch']
		loss = checkpoint['loss']

		print("Model resumed for training...")
		print("Epoch: ", epoch_resume)
		print("Loss: ", loss)

	print("Loaded checkpoint from: {}".format(path))
	return model, epoch_resume



class SpeakerEncoder(nn.Module):
	def __init__(self, mel_n_channels, model_hidden_size, model_num_layers, model_embedding_size):
		super().__init__()
		
		# Network defition
		self.lstm = nn.LSTM(input_size=mel_n_channels,
							hidden_size=model_hidden_size, 
							num_layers=model_num_layers, 
							batch_first=True)
		self.linear = nn.Linear(in_features=model_hidden_size, 
								out_features=model_embedding_size)
		self.relu = torch.nn.ReLU()
			   
		# Cosine similarity scaling (with fixed initial parameter values)
		self.similarity_weight = nn.Parameter(torch.tensor([10.]))
		self.similarity_bias = nn.Parameter(torch.tensor([-5.]))
	
	def forward(self, utterances, hidden_init=None):
		"""
		Computes the embeddings of a batch of utterance spectrograms.
		
		:param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
		(batch_size, n_frames, n_channels) 
		:param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
		batch_size, hidden_size). Will default to a tensor of zeros if None.
		:return: the embeddings as a tensor of shape (batch_size, embedding_size)
		"""
		# Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
		# and the final cell state.
		out, (hidden, cell) = self.lstm(utterances, hidden_init)
		
		# We take only the hidden state of the last layer
		embeds_raw = self.relu(self.linear(hidden[-1]))
		
		# L2-normalize it
		embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
		
		return embeds





speaker_enc = SpeakerEncoder(hp.mel_n_channels, hp.model_hidden_size, hp.model_num_layers, hp.model_embedding_size)

for p in speaker_enc.parameters():
		p.requires_grad = False
from pathlib import Path
cwd = Path.cwd()
voicedisc_checkpoint_path=os.path.join(cwd,"speaker_embedding","saved_models/checkpoint_step00335.pt")
print("path:",voicedisc_checkpoint_path)
speaker_enc, _ = load_checkpoint(voicedisc_checkpoint_path, speaker_enc, None, reset_optimizer=True, disc=True)



