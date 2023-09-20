# from tensorflow.contrib.training import HParams
from glob import glob
import os, pickle
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

'''
def _get_image_list(dataset, split, path):
    pkl_file = 'filenames_{}_{}.pkl'.format(dataset, split)
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as p:
            return pickle.load(p)
    else:
        filelist = glob(path)
        random.shuffle(filelist)
        
        if split == 'train':
            filelist = filelist[:int(.95 * len(filelist))]
        else:
            filelist = filelist[int(.95 * len(filelist)):]

        with open(pkl_file, 'wb') as p:
            pickle.dump(filelist, p, protocol=pickle.HIGHEST_PROTOCOL)

        return filelist

def _get_files_lrs2(split, path):
    fname = 'scripts/filelists/{}.txt'.format(split)
    files = np.loadtxt(fname, str)

    filelist = []
    for i in range(len(files)):
        filelist.append(os.path.join(path, files[i]))

    return filelist

def _get_filelist_lrw(dataset, split, path):
    pkl_file = 'filenames_{}_{}.pkl'.format(dataset, split)
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as p:
            filelist = pickle.load(p)
        # print(len(filelist))
        return filelist
    else:
        filelist = glob(path)
        print(len(filelist))
        
        with open(pkl_file, 'wb') as p:
            pickle.dump(filelist, p, protocol=pickle.HIGHEST_PROTOCOL)

        return filelist

def _get_all_files(split):

    # print("Split: ", split)

    # LRW train files
    # filelist_lrw = _get_image_list('lrw', split, '/ssd_scratch/cvit/sindhu/preprocessed_lrw/*/*/*')
    # print("LRW: ", len(filelist_lrw))

    # LRS2 train files
    filelist_lrs2 = _get_files_lrs2(split, '/ssd_scratch/cvit/sindhu/preprocessed_lrs2_train')
    # print("LRS2: ", len(filelist_lrs2))

    # LRS2 pre-train files
    filelist_lrs2_pretrain = _get_image_list('lrs2_pretrain', split, '/ssd_scratch/cvit/sindhu/preprocessed_lrs2_pretrain/*/*')
    # print("LRS2 pre-train: ", len(filelist_lrs2_pretrain))

    # LRS3 train files
    # filelist_lrs3 = _get_image_list('lrs3', split, '/ssd_scratch/cvit/sindhu/preprocessed_lrs3_train/*/*')
    # print("LRS3: ", len(filelist_lrs3))

    # LRS3 pre-train files
    # filelist_lrs3_pretrain = _get_image_list('lrs3_pretrain', split, '/ssd_scratch/cvit/prajwal/preprocessed_lrs3_pretrain/*/*')
    # print("LRS3 pre-train: ", len(filelist_lrs3_pretrain))

    # Combine all the files
    # filelist = filelist_lrs2 + filelist_lrs2_pretrain + filelist_lrs3 + filelist_lrs3_pretrain
    filelist = filelist_lrs2 + filelist_lrs2_pretrain

    # print("------------------------------------------------------------------------------------")

    return filelist

def _get_filelist(split):

    if split=='train':
        # print("Train:")
        filelist_lrw = _get_filelist_lrw('lrw', 'train', '/ssd_scratch/cvit/sindhu/preprocessed_lrw/*/*/*')
        filelist_lrs2 = _get_files_lrs2('train', '/ssd_scratch/cvit/sindhu/preprocessed_lrs2_train')
    else:
        # print("Val:")
        filelist_lrw = _get_filelist_lrw('lrw', 'val', '/ssd_scratch/cvit/sindhu/preprocessed_lrw_val/*/*/*')
        filelist_lrs2 = _get_files_lrs2('val', '/ssd_scratch/cvit/sindhu/preprocessed_lrs2_train')

    # print("LRW: ", len(filelist_lrw))
    # print("LRS2: ", len(filelist_lrs2))
    filelist = filelist_lrw + filelist_lrs2

    # print("Filelist: ", len(filelist))
    # print("------------------------------------------------------------------------------------")

    return filelist
'''

class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value
        
hparams = HParams(

    num_mels=80, 
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, 
    # also consider clipping your samples to smaller chunks)
    max_mel_frames=900,
    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3
    #  and still getting OOM errors.
    
    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,
    # use_lws=True.
    
    n_fft=512,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=160,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=400,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)

    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    
    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    
    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
    # be too big to avoid gradient explosion, 
    # not too small for fast convergence)
    normalize_for_wavenet=True,
    # whether to rescale to [0, 1] for wavenet. (better audio quality)
    clip_for_wavenet=True,
    # whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
    
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.
    
    # Griffin Lim
    power=1.5,
    # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters=60,
    # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
    ###########################################################################################################################################
    
    # Model params
    builder='nyanko',
    downsample_step=4,
    max_positions=512,
    binary_divergence_weight=0.,
    priority_freq=3000,
    use_guided_attention=True,
    guided_attention_sigma=0.2,

    N=25,
    frame_overlap=0,
    mel_overlap=0,
    start_idx=0,
    mel_start_idx=0,
    img_size=96,
    fps=25,
    spec_step_size=100,
    wav_step_size=16000,

    # all_images_lrw=_get_filelist_lrw('lrw', 'train', '/ssd_scratch/cvit/sindhu/preprocessed_lrw/*/*/*'),
    # all_test_images_lrw=_get_filelist_lrw('lrw', 'val', '/ssd_scratch/cvit/sindhu/preprocessed_lrw_val/*/*/*'),

    # all_images_lrs2=_get_files_lrs2('train', '/ssd_scratch/cvit/sindhu/preprocessed_lrs2_train'),
    # all_test_images_lrs2=_get_files_lrs2('val', '/ssd_scratch/cvit/sindhu/preprocessed_lrs2_train'),

    # all_images=_get_filelist('train'),
    # all_test_images=_get_filelist('val'),
    
    # all_images_fulldata=_get_all_files('train'),
    # all_test_images_fulldata=_get_all_files('val'),

    # lrw_val_files=glob('/ssd_scratch/cvit/sindhu/preprocessed_lrw_test/PROBLEMS/*/*'),

    #all_images_grid=_get_image_list('grid', 'train', '/ssd_scratch/cvit/sindhu/preprocessed_grid_train/*/*'),
    #all_test_images_grid=_get_image_list('grid', 'val', '/ssd_scratch/cvit/sindhu/preprocessed_grid_train/*/*'),

    #all_images_timit=_get_image_list('timit', 'train', '/ssd_scratch/cvit/sindhu/preprocessed_TIMIT/*/*'),
    #all_test_images_timit=_get_image_list('timit', 'val', '/ssd_scratch/cvit/sindhu/preprocessed_TIMIT/*/*'),
    
    
    resume=True,
    #checkpoint_dir = '/scratch/prajwalkr_rudra/syncnet_lrs2_noisy_checkpoints/',
    #checkpoint_path='checkpoints/color_corrected_lipgan_checkpoint_step396k.pth',

    n_gpu=1,
    batch_size=16,
    adam_beta1=0.5,
    adam_beta2=0.9,
    adam_eps=1e-6,
    amsgrad=False,
    initial_learning_rate=1e-4,
    disc_initial_learning_rate=1e-4,
    lr_schedule=None,#"noam_learning_rate_decay",
    lr_schedule_kwargs={},
    nepochs=200000000000000000,
    weight_decay=0.0,
    clip_thresh=0.001,
    num_workers=1,
    checkpoint_interval=3000,
    eval_interval=3000,
    save_optimizer_state=True,
    gradient_penalty_wt=10,
    gen_interval=5,

    syncnet_wt=0.3,
    l1_wt=10,
    kl_wt=5,
    voice_wt=5,
    unit_kl_wt=2,
    local_kl_wt=5,

    syncnet_wt_higher=1,
    l1_wt_higher=25,
    kl_wt_higher=20,
    local_kl_wt_higher=20,
    voice_wt_higher=10,

    n_class=500,
    n_class_finetune=1628,

    ## Speaker emb model parameters
    mel_n_channels = 80,
    model_hidden_size = 256,
    model_embedding_size = 256,
    model_num_layers = 3,

    num_segments=10

)


def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)




