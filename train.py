# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import argparse
import multiprocessing
import os
from tqdm import tqdm 
from audio_features.hparams import hparams
from model_with_linear import Speech_Enhancer
from data import NoisyCleanPair

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import yaml
from speaker_embedding.speaker_embed import speaker_enc


from audio_features.audio import save_wav
from utils import convert_spec_to_wav, convert_mel_to_wav
from losses import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, train_loss, checkpoint_dir, epoch):
    
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_step{:04d}.pt".format(epoch))

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": train_loss,
        "epoch": epoch,
    }, checkpoint_path)
    
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    
    new_s = {}

    for k, v in s.items():
        if torch.cuda.device_count()>1:
            if not k.startswith('module.'):
                new_s['module.'+k] = v
            else:
                new_s[k] = v
        else:
            new_s[k.replace('module.', '')] = v
    
    model.load_state_dict(new_s)

    epoch_resume = 0
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])

        epoch_resume = checkpoint['epoch']
        #loss = checkpoint['loss']

        print("Model resumed for training...")
        
    
    return model, optimizer, epoch_resume

def train(context_enc, speaker_enc, model, train_loader, args, cfg, logdir, device, iteration):

    total_batch = len(train_loader)
    print("Total train batch: ", total_batch)
    
    
    total_loss=0
    context_embed, speaker_embed=[], []

    ite=iteration
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_resume+1, cfg['train']['num_epochs']):

        progress_bar = tqdm(enumerate(train_loader))
        total_loss = 0.0

        print("Epoch: {}".format(epoch))


        for  step, ((mel_n_batch, mel_n, gt_mel_c), (noisy_speech, gt_clean_speech), (linear_noisy, gt_linear_clean)) in progress_bar:#iter(train_loader):#
            
            model.train()
            optimizer.zero_grad()

            
            prosody_embed=torch.unsqueeze(speaker_enc(mel_n_batch[0].to(device)),dim=1)
            
           
            prosody_embed=torch.cat([prosody_embed,prosody_embed], dim=1)
            
            for s in range(1, len(mel_n_batch)):
                
                spec=mel_n_batch[s]
                
                spk_embed = torch.unsqueeze(speaker_enc(spec.to(device)),dim=1)
                
                for i in range(2):
                    prosody_embed=torch.cat([prosody_embed,spk_embed], dim=1)

                
            # Speech Enhancer
            rec_clean_linear_spec,  r_mel=model(prosody_embed, linear_noisy[:,:,:-1].to(device), mel_n.to(device))
           
            # loss
            rec_loss = get_reconstruction_loss(rec_clean_linear_spec, gt_linear_clean[:,:,:-1].to(device))
            rec_loss_mel = get_reconstruction_loss(r_mel, gt_mel_c.to(device))
            
            speaker_loss = get_speaker_embed_loss(r_mel, gt_mel_c, speaker_enc, device)
            
            loss=10*rec_loss + 5*rec_loss_mel + 1*speaker_loss  

            total_loss += loss

            loss.backward()
            optimizer.step()

            ite+=1
            # Display the training progress
            progress_bar.set_description('ite: {}, Loss: {}'.format(ite, total_loss / (step + 1))) 
            
            progress_bar.refresh()


            if ite % 500== 0:
                save_checkpoint(model, optimizer, total_loss, checkpoint_dir, ite)

            
            del mel_n_batch, gt_mel_c, noisy_speech, gt_clean_speech, linear_noisy, gt_linear_clean


        train_loss = total_loss / total_batch

        print("train_loss:",train_loss)
        
        
# Function to generate and save the sample audio/video files
def save_samples(gt_linear_clean, gt_mel_c, rec_clean_linear_spec, r_mel, epoch_i, checkpoint_dir):

    r_linear_clean = rec_clean_linear_spec.detach().cpu().numpy()
    b, s, f= rec_clean_linear_spec.shape
    temp=np.zeros([b, s, f+1])
    temp[:,:,:-1]=rec_clean_linear_spec[:,:,:]
    r_wavs=convert_spec_to_wav(temp, hparams)
    wavs=convert_spec_to_wav(gt_linear_clean, hparams)

    
    path=os.path.join(checkpoint_dir, "output", str(epoch_i))
    os.system("mkdir -p "+path)
    
    save_wav(r_wavs, 16000, os.path.join(path, "r_wav.wav"))
    save_wav(wavs, 16000, os.path.join(path, "gt_wav.wav"))

  




def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--case', required=False, type=str, default="net1", help='experiment case name')
    parser.add_argument('-cfg','--config_file', default="config.yaml", type=str, help='location of config file')
    parser.add_argument('--checkpoint_dir', default="/scratch/jaya/chkpts",required=False, type=str, help='Folder to save the model')
    parser.add_argument('--checkpoint_path', default="", type=str, help='Path of the saved model to resume training')
    arguments = parser.parse_args()

    return arguments





if __name__ == '__main__':
    args = get_arguments()
    print("args")

    f = open(args.config_file, 'r')
    cfg = list(yaml.load_all(f, Loader=yaml.FullLoader))[0]


    logdir_train = '{}/train'.format(cfg['logdir_path'])
    
    c_model=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
    
    if torch.cuda.device_count()>1:
        speaker_enc = nn.DataParallel(speaker_enc)
    
    context_enc=c_model.to(device)

    
    speaker_enc.to(device)
    

    print("speech_Root:",cfg['train']['speech_root'])

    train_dataset = NoisyCleanPair(cfg['train']['speech_root'], cfg['train']['noise_root'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, 
        drop_last=False, num_workers=cfg['train']['num_jobs'])
      
    model = Speech_Enhancer(d_model_spk=cfg['train']['d_model_spk'],
                n_enc_heads=cfg['train']['n_enc_heads'], n_enc_attn=cfg['train']['n_enc_attn'],  
                n_out_seq=cfg['train']['n_out_seq'], n_dec_heads=cfg['train']['n_dec_heads'], dec_d_model=512,
                n_dec_attn=cfg['train']['n_dec_attn'], reduction_factor=cfg['train']['reduction_factor'], 
                dropout=cfg['train']['dropout'], output_mel=cfg['train']['d_output_mel'], output_linear=256,
        )

    
    if torch.cuda.device_count()>1:
        print("Using", torch.cuda.device_count(), "GPUs for the denoising model!")
        model = nn.DataParallel(model)

    model=model.to(device)
    
    # Set the learning rate
    if cfg['train']['reduced_learning_rate'] is not None:
        lr = cfg['train']['reduced_learning_rate']
    else:
        lr = cfg['train']['lr']
    print("learning_rate:",cfg['train']['lr'])
    
    optimizer = Adam( model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)    


    iteration=0

    
    if args.checkpoint_path is not None:
        model, optimizer, iteration = load_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint_path), model, optimizer)
        print("chkpt is loaded")    
    

    # Create the folder to save checkpoints
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    train(context_enc, speaker_enc, model, train_loader, args, cfg, logdir_train, device, iteration)



    print("Traininig completed")
