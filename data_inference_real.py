import argparse
import torch 
import numpy as np

import random
from audio_features.audio import *
from glob import glob
from audio_features.hparams import hparams
import random
import numpy as np

class NoisyCleanPair:
    def __init__(self, speech_root,  augBool=True):
        """__init__.

        """
        
        #self.noise_wavs=glob(noise_root+"/*.wav")
        self.speech_wavs=glob(speech_root+"/*.mp4")
        
        self.SNR=range(-5, 5, 1)
        self.augBool=augBool

        print("speech root:",speech_root)
    def __len__(self):
        return 1000#len(self.speech_wavs)

    def __getitem__(self, index):

        while(1):

            #noise_idx=random.sample(range(len(self.noise_wavs)), 1)[0]
            speech_idx=random.sample(range(len(self.speech_wavs)), 1)[0]
            print("wav:",self.speech_wavs[speech_idx])
            speech=load_wav(self.speech_wavs[speech_idx], hparams.sample_rate)
            #noise=load_wav(self.noise_wavs[noise_idx], hparams.sample_rate)


            print("speech:loaded")

            if len(speech)<hparams.sample_rate:
                print("speech={}, noise={}, hparams.sample_rate={}".format(len(speech), noise.shape, hparams.sample_rate))
                continue

        
            s=random.sample(range(len(speech)-hparams.sample_rate),1)[0]
            mel_n_batch_chunks, mel_c_chunks, mel_n_chunks, noisy_speech_chunks, speech_crop_chunks, linear_noisy_chunks, linear_clean_chunks=[], [], [], [], [], [], []
            print("len(speech):{}, sample_rate={}".format(len(speech)-1, hparams.sample_rate))
            chunks=(len(speech)-1)/ hparams.sample_rate
            print("chunks:",chunks)
            for i in range(0, len(speech)-1, hparams.sample_rate):
                noisy_speech=speech[i:i+hparams.sample_rate]

                
                

                stft_n, mel_n, linear_n = all_spec(noisy_speech, hparams)
                
                if np.sum(mel_n) is None \
                    or  np.sum(linear_n) is None \
                    or np.sum(noisy_speech) is None :
                    print("condition: np.sum(mel_n) is None :", (np.sum(mel_n) is None) )
                    continue

                wave_slices, mel_slices = self.compute_overlapped_mel_slices(len(noisy_speech), mel_samples=len(mel_n.T), mels_per_sec=100, target_frames=25, word_per_sec=4)
                
                
                mel_n, linear_noisy = mel_n.T, linear_n.T
                mel_n_batch = [torch.FloatTensor(np.array(mel_n[s])) for s in mel_slices]
                print("mel_n_batch:",len(mel_n_batch))
                mel_n_batch_chunks.append(mel_n_batch)
                mel_n_chunks.append(torch.FloatTensor(mel_n[:-1, :]))
               
                noisy_speech_chunks.append(noisy_speech)
                
                linear_noisy_chunks.append(torch.FloatTensor(linear_noisy[:-1,:]))
                
            print("mel_n_batch_chunks:",len(mel_n_batch_chunks),"mel_n_chunks:",len(mel_n_chunks))
            return (mel_n_batch_chunks, mel_n_chunks), (noisy_speech_chunks), (linear_noisy_chunks)
        
    '''
    def __getitem__(self, index):

        while(1):

            noise_idx=100#random.sample(range(len(self.noise_wavs)), 1)[0]
            speech_idx=100#random.sample(range(len(self.speech_wavs)), 1)[0]
            
            speech=load_wav(self.speech_wavs[speech_idx], hparams.sample_rate)
            noise=load_wav(self.noise_wavs[noise_idx], hparams.sample_rate)

            
            mel_n_batch_chunks, mel_c_chunks, noisy_speech_chunk, speech_crop_chunk=[], [], [], []
            linear_noisy_chunk, linear_clean_chunk = [], []

            for s in range(0, len(speech), hparams.sample_rate):
                speech_crop=speech[s:s+hparams.sample_rate]
                

                            
                                
                if len(noise)<=len(speech_crop):
                    print("condition --- len(noise)<=len(speech_crop) :",len(noise)<=len(speech_crop))
                    continue
                t=random.sample(range(len(noise)-len(speech_crop)),1)[0]
                noise_crop=noise[t:t+len(speech_crop)]

                
                #add noise
                snr=0#random.sample(self.SNR,1)[0]
                noisy_speech=self.addNoise(speech_crop, noise_crop, snr)

               

                stft_n, mel_n, linear_n = all_spec(noisy_speech, hparams)
                stft_c, mel_c, linear_c = all_spec(speech_crop, hparams)
                #print("stft_c : {}, mel_c : {}, linear_c : {}".format(stft_c.shape, mel_c.shape, linear_c.shape))
                if np.sum(mel_c) is None or np.sum(mel_n) is None \
                    or np.sum(linear_c) is None or np.sum(linear_n) is None \
                    or np.sum(noisy_speech) is None or np.sum(speech_crop) is None:
                    print("condition: np.sum(mel_c) is None :", (np.sum(mel_c) is None) )
                    continue

                wave_slices, mel_slices = self.compute_overlapped_mel_slices(len(noisy_speech), mel_samples=len(mel_n.T), mels_per_sec=100, target_frames=25, word_per_sec=4)
                
                
                mel_n, linear_noisy, linear_clean=mel_n.T, linear_n.T, linear_c.T
                mel_n_batch = [torch.FloatTensor(np.array(mel_n[s])) for s in mel_slices]

                
                mel_n_batch_chunks.append(mel_n_batch)
                mel_c_chunks.append(torch.FloatTensor(mel_c.T[:-1, :]))

                noisy_speech_chunk.append(noisy_speech)
                speech_crop_chunk.append(speech_crop)

                linear_noisy_chunk.append(linear_noisy[:-1,:])
                linear_clean_chunk.append(linear_clean[:-1,:])

            return (mel_n_batch_chunks, mel_c_chunks), (noisy_speech_chunk, speech_crop_chunk), (linear_noisy_chunk, linear_clean_chunk)
        
        '''
    

    def cal_rms(self, amp):
        return np.sqrt(np.mean(np.square(amp), axis=-1))

    def cal_adjusted_rms(self, clean_rms, snr):
        a = float(snr) / 20
        noise_rms = clean_rms / (10**a) 
        return noise_rms 

    '''

    def add_Noise(self, gt_wav, random_wav, desired_snr):

        samples = len(gt_wav)

        signal_power = np.sum(np.square(np.abs(gt_wav)))/samples
        noise_power = np.sum(np.square(np.abs(random_wav)))/samples

        k = (signal_power/(noise_power+1e-8)) * (10**(-desired_snr/10))

        scaled_random_wav = np.sqrt(k)*random_wav

        noisy_wav = gt_wav + scaled_random_wav

        return noisy_wav
    '''
    def addNoise(self, speech, noise, snr):
        speech_rms=self.cal_rms(speech)
        noise_rms=self.cal_rms(noise)
        adjusted_noise_rms = self.cal_adjusted_rms(speech_rms, snr)
        adjusted_noise_amp = noise * (adjusted_noise_rms / (noise_rms+0.0001)) 
        mixed_amp = (speech + adjusted_noise_amp)

        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min

        if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
            if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
                reduction_rate = max_int16 / mixed_amp.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed_amp.min(axis=0)
            mixed_amp = mixed_amp * (reduction_rate)
            
        return mixed_amp
    

    def compute_overlapped_mel_slices(self, wav_samples, mel_samples, mels_per_sec, target_frames, word_per_sec=4):
        samples_per_frame_in_wav = np.int(hparams.sample_rate/word_per_sec)
        samples_per_frame_in_mel = np.int(mels_per_sec/word_per_sec)
        overlap_frames_mel= np.int((mel_samples-target_frames)/(samples_per_frame_in_mel-1))+1
        

        overlap_frames_wav=np.int((wav_samples-target_frames)/(samples_per_frame_in_wav-1))+1

        

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

    def augmentation(self, gt_wav, noisy_seg_wav, gt_seg_wav):

        aug_steps = np.random.randint(low=0, high=3200)
        aug_start_idx = np.random.randint(low=0, high=hparams.sample_rate - aug_steps)
        aug_end_idx = aug_start_idx+aug_steps

        aug_types = ['No', 'zero_speech', 'reduce_speech', 'increase_noise']
        aug = random.choice(aug_types)
        

        if aug == 'zero_speech':    
            noisy_seg_wav[aug_start_idx:aug_end_idx] = 0.0
            
        elif aug == 'reduce_speech':
            noisy_seg_wav[aug_start_idx:aug_end_idx] = 0.1*gt_seg_wav[aug_start_idx:aug_end_idx]

        elif aug == 'increase_noise':
            # Load the random noisy wav file
            _idx=random.sample(range(len(self.noise_wavs)), 1)[0]

            random_wav = load_wav(self.noise_wavs[_idx], hparams.sample_rate)

            while len(random_wav)<=aug_end_idx:
                _idx=random.sample(range(len(self.noise_wavs)), 1)[0]
                random_wav = load_wav(self.noise_wavs[_idx], hparams.sample_rate)
                print("inside augmentation loop")

            noisy_seg_wav[aug_start_idx:aug_end_idx] = gt_seg_wav[aug_start_idx:aug_end_idx] + (2*random_wav[aug_start_idx:aug_end_idx])

        return noisy_seg_wav

    

