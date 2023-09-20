import torch
import torch.nn as nn
import torch.nn.functional as F
from speaker_embedding.speaker_embed import speaker_enc

def get_reconstruction_loss(generated_spec, gt_spec):

	l1 = nn.L1Loss()
	
	l1_loss = l1(generated_spec, gt_spec)

	return l1_loss

def get_orthogonality_loss(feat_1, feat_2):

	# Orthogonality loss = lambda*(1-(A.B)/|A|.|B|)^2) : lambda is a regularization term, A and B are features

	# The basic idea : two vector space is ortho normal to each other when  each vector from one vector-space 
	#				   is ortho-normal to any vector in the other vector-space

	

	feat_1=F.relu(feat_1)
	feat_2=F.relu(feat_2)


	_, _, embed_dim=feat_1.shape

	feat_1=feat_1.view(-1, embed_dim)
	feat_2=feat_2.view(-1, embed_dim)

	mod_feat_1=torch.norm(feat_1, dim=1)
	mod_feat_2=torch.norm(feat_2, dim=1)

	#print("mod_feat_1:",mod_feat_1.shape)

	denominator=torch.mul(torch.unsqueeze(mod_feat_1, dim=1), mod_feat_2)

	#print("denominator:",denominator.shape)

	dot_prod=torch.sum(torch.mul(torch.unsqueeze(feat_1, dim=1), feat_2), dim=2)

	#print("dot_prod:",dot_prod.shape)

	orthogonality=torch.div(dot_prod, denominator)

	#print("orthogonality:",orthogonality.shape)

	#print("average:",torch.mean(orthogonality))
	
	disentangle_loss=torch.square(1-torch.mean(orthogonality))
	
	#print("disentangle_loss:",disentangle_loss)

	return disentangle_loss

def get_wav2vec2_loss(rec_feat, clean_feat):
	return get_reconstruction_loss(rec_feat, clean_feat)

def get_speaker_embed_loss(r_mel, clean_mel, model, device):

	#print("r_mel={}, clean_mel:{}".format(r_mel.shape, clean_mel.shape))

	c_feat=model(clean_mel.to(device))
	r_feat=model(r_mel.to(device))
	
	

	#print("c_feat:{}".format(c_feat.shape))
	#print("r_feat:", r_feat.shape)

	return get_reconstruction_loss(r_feat, c_feat)
