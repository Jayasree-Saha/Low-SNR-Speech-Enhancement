from __future__ import print_function

import argparse
import multiprocessing
import os
from tqdm import tqdm 
from audio_features.hparams import hparams
#from model import Speech_Enhancer
from model_with_linear import Speech_Enhancer
from data_inference_real import NoisyCleanPair

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

from audio_features.audio import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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


def load_checkpoint(path, model):

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    #print("checkpoint:",checkpoint["state_dict"].keys())
    s = checkpoint["state_dict"]
    
    new_s = {}

    for k, v in s.items():
        if 'proj_dim' in k or 'upsample_ctxt' in k:
            continue
        if torch.cuda.device_count() > 1:
            
            if not k.startswith('module.'):
                new_s['module.'+k] = v
            else:
                new_s[k] = v
        else:
            new_s[k.replace('module.', '')] = v
    #print("new_s:",s.keys()) 
    
    model.load_state_dict(new_s)
    '''
    epoch_resume = 0
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    
        #epoch_resume = checkpoint['epoch']
        #loss = checkpoint['loss']

        print("Model resumed for training...")
        #print("Epoch: ", epoch_resume)
        #print("Loss: ", loss)
    '''
    return model#, optimizer#, epoch_resume

def infer(speaker_enc, model, dataset):

        
        is_overflow = False
        # ================ MAIN TRAINNIG LOOP! ===================
        total_loss=0
        context_embed, speaker_embed=[], []

       
        
        mel_n_batch_chunks, mel_n_chunks, noisy_speech_chunks, linear_noisy_chunks=dataset
        results=[]
        
        for  i in range(len(mel_n_batch_chunks)-1):
            #print("gt_mel_c_chunks:",gt_mel_c_chunks[0].shape)
            
            
            mel_n_batch, mel_n = mel_n_batch_chunks[i], mel_n_chunks[i]
            noisy_speech = noisy_speech_chunks[i]
            linear_noisy = linear_noisy_chunks[i]

            # wav2vec 2.0 feature and prosody feature from speacker embedding
            
            #print("mel_n_batch[0]:",mel_n_batch[0].shape)
            mel_n_batch[0]=torch.unsqueeze(mel_n_batch[0], 0)
            #print("mel_n_batch[0]:",mel_n_batch[0].shape)
            
            prosody_embed=torch.unsqueeze(speaker_enc(mel_n_batch[0].to(device)),dim=1)
            
           
            prosody_embed=torch.cat([prosody_embed,prosody_embed], dim=1)

            for s in range(1, len(mel_n_batch)):
                
                spec=mel_n_batch[s]
                spec=torch.unsqueeze(spec, 0)
                #print("spk:",speaker_enc(spec.to(device)).shape)
                spk_embed = torch.unsqueeze(speaker_enc(spec.to(device)),dim=1)
                
                for i in range(2):
                    prosody_embed=torch.cat([prosody_embed, spk_embed], dim=1)
            linear_noisy = torch.unsqueeze(linear_noisy,dim=0) 
            #print("prosody_embedding:",prosody_embed.shape)
            #print("linear_noisy:",linear_noisy.shape)
            rec_clean_linear_spec,  r_mel=model(prosody_embed, linear_noisy[:,:,:-1].to(device), mel_n.to(device))

           
            #print("mel loss: ", get_reconstruction_loss(r_mel, gt_mel_c.to(device)))
            #print("spec loss: ", get_reconstruction_loss(rec_clean_linear_spec, gt_linear_clean[:,:,:-1].to(device)))

            results.append([rec_clean_linear_spec, noisy_speech, r_mel])
        save_samples(results)#, noisy_file, predicted_file)
        print("done")
        #break
        
            

def save_samples(results):#, noisy_file, predicted_file):
    long_r_spec, long_noisy_spec, long_r_mel=[], [], []
    for i in range(len(results)):
        rec_clean_linear_spec, noisy_speech, r_mel=results[i]
        r_linear_clean = rec_clean_linear_spec.detach().cpu().numpy()
        #print("r_mel:",r_mel.shape)
        r_mel_clean= r_mel.detach().cpu().numpy()
        b, s, f= r_linear_clean.shape
        temp=np.zeros([b, s, f+1])
        temp[:,:,:-1]=r_linear_clean[:,:,:]
        spec=temp[0, :, :].T
        r_wav=inv_linear_spectrogram(spec,hparams)
        r_mel_wav=inv_mel_spectrogram(np.squeeze(r_mel_clean).T, hparams)
        #print("r_wav:",r_wav.shape)
        long_r_spec.append(r_wav)
        noisy_speech=np.squeeze(noisy_speech)#.numpy()
        #gt_clean_speech=np.squeeze(gt_clean_speech.numpy())
        long_noisy_spec.append(noisy_speech)
        #long_gt_spec.append(gt_clean_speech)
        long_r_mel.append(r_mel_wav)


    long_r_spec=np.concatenate(long_r_spec, axis=0)
    #long_gt_spec=np.concatenate(long_gt_spec, axis=0)
    #save_wav(long_gt_spec, 16000,  "gt_wav.wav")
    save_wav(long_r_spec, 16000,  "videoplayback_r_wav.wav")

        
        
    print("Saved samples done:")
def compute_overlapped_mel_slices(wav_samples, mel_samples, mels_per_sec, target_frames, word_per_sec=8):
        samples_per_frame_in_wav = int(hparams.sample_rate/word_per_sec)
        samples_per_frame_in_mel = int(mels_per_sec/word_per_sec)
        #print("samples_per_frame_in_mel:",samples_per_frame_in_mel)
        overlap_frames_mel= int((mel_samples-target_frames)/(samples_per_frame_in_mel-1))+1
        

        overlap_frames_wav=int((wav_samples-target_frames)/(samples_per_frame_in_wav-1))+1

        

        wav_slices, mel_slices = [], []
        start, start_w=0, 0
        for i in range(target_frames-1):

            mel_range=np.array([start,start+samples_per_frame_in_mel])
            mel_slices.append(slice(*mel_range))
            start=start+overlap_frames_mel


            wav_range=np.array([start,start+samples_per_frame_in_wav])
            wav_slices.append(slice(*wav_range))
            start_w=start_w+overlap_frames_wav


          
        mel_range=np.array([start,mel_samples])
        mel_slices.append(slice(*mel_range))
        

        wav_range=np.array([start,start+samples_per_frame_in_wav])
        wav_slices.append(slice(*wav_range))
    
        return wav_slices, mel_slices  
def get_dataset(file_name):
   speech=load_wav(file_name, hparams.sample_rate)
   mel_n_batch_chunks,  mel_n_chunks, noisy_speech_chunks,  linear_noisy_chunks=[], [],  [],   [] 
   
   for i in range(0, len(speech), hparams.sample_rate):
       noisy_speech=speech[i:i+hparams.sample_rate]
       stft_n, mel_n, linear_n = all_spec(noisy_speech, hparams)
       wave_slices, mel_slices = compute_overlapped_mel_slices(len(noisy_speech), mel_samples=len(mel_n.T), mels_per_sec=100, target_frames=25, word_per_sec=2)
       
       mel_n, linear_noisy = mel_n.T, linear_n.T
       mel_n_batch = [torch.FloatTensor(np.array(mel_n[s])) for s in mel_slices]
       
       mel_n_batch_chunks.append(mel_n_batch)
       mel_n_chunks.append(torch.FloatTensor(mel_n[:-1, :]))
       
       noisy_speech_chunks.append(noisy_speech)
       linear_noisy_chunks.append(torch.FloatTensor(linear_noisy[:-1,:]))
       
   return mel_n_batch_chunks, mel_n_chunks, noisy_speech_chunks, linear_noisy_chunks
   
parser = argparse.ArgumentParser()
parser.add_argument("-c", '--case', required=False, type=str, default="net1", help='experiment case name')
parser.add_argument('-cfg','--config_file', default="config.yaml", type=str, help='location of config file')
parser.add_argument('--checkpoint_dir', default="/ssd_scratch/cvit/jaya/",required=False, type=str, help='Folder to save the model')
parser.add_argument('--checkpoint_path', default="checkpoint_step204000.pt", type=str, help='Path of the saved model to resume training')

parser.add_argument('--ckpt_freq', default=1, required=False, type=str, help='checkpoint to load model.')
parser.add_argument('--continue_epoch', default=True, help='Continue epoch number?')
    
args = parser.parse_args()

f = open(args.config_file, 'r')
cfg = list(yaml.load_all(f, Loader=yaml.FullLoader))[0]
logdir_train = '{}/train'.format(cfg['logdir_path'])

   
    
speaker_enc.to(device)

model = Speech_Enhancer(d_model_spk=cfg['train']['d_model_spk'],
                n_enc_heads=cfg['train']['n_enc_heads'], n_enc_attn=cfg['train']['n_enc_attn'],  
                n_out_seq=cfg['train']['n_out_seq'], n_dec_heads=cfg['train']['n_dec_heads'], dec_d_model=512,
                n_dec_attn=cfg['train']['n_dec_attn'], reduction_factor=cfg['train']['reduction_factor'], 
                dropout=cfg['train']['dropout'], output_mel=cfg['train']['d_output_mel'], output_linear=256,
)
        
model=model.to(device)

model = load_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint_path), model)
print("chkpt is loaded")   
model.eval()


dataset=get_dataset("videoplayback_v1.wav")
infer(speaker_enc, model,  dataset)
     
 
