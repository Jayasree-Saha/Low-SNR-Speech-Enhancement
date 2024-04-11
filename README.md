
# Speech Enhancement in Low SNR Environments

Official repository for our paper, "An Approach for Speech Enhancement in Low SNR Environments using Granular Speaker Embedding", CODS-COMAD 2024.





[![License](https://img.shields.io/badge/License-CC_BY_NC_4.0-green.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Paper arxiv](https://img.shields.io/badge/Paper-CVIT_IIIT_Hyderabad-yellow.svg)](https://cvit.iiit.ac.in/images/ConferencePapers/2024/3632410.3632413.pdf)


## Authors

[Jayasree Saha](https://github.com/Jayasree-Saha), [Rudrabha Mukhopadhyay](https://rudrabha.github.io/), Aparna Agrawal, Surabhi Jain, and  [CV Jawahar](https://faculty.iiit.ac.in/~jawahar/) 


## Abstract
The proliferation of speech technology applications has led to an un-
precedented demand for effective speech enhancement techniques,
particularly in low Signal-to-Noise Ratio (SNR) conditions. This
research presents a novel approach to speech enhancement, specif-
ically designed for very low SNR scenarios. Our technique focuses
on speaker embedding at a granular level and highlights its consis-
tent impact on enhancing speech quality and improving Automatic
Speech Recognition (ASR) performance, a significant downstream
task. Experimental findings demonstrate competitive speech qual-
ity and substantial enhancements in ASR accuracy compared to
alternative methods in low SNR situations. The proposed technique
offers promising advancements in addressing the challenges posed
by low SNR conditions in speech technology applications.


## Demo

To be uploaded soon!!!


## Prerequisites
- Python 3.8.18  (Code has been tested with this version)
- Install required packages using the following command
```bash
  pip install -r requirements.txt
```

## Deployment
We illustrate the training process using the LRS3 and VGGSound dataset. Adapting for other datasets would involve small modifications to the code.


### Preprocess the dataset
#### LRS3 pre-train dataset folder structure
```bash
data_root (we use only  pre-train sets of LSR3 dataset in this work)
├── list of folders
│   ├── video IDs ending with (.mp4) (We convert .mp4 to .wav before data reading)

```

#### VGGSound folder structure
We use VGGSound dataset as noisy data which is mixed with the clean speech from LRS3 dataset. We download the audio files (*.wav files) from [here](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)

#### Training

```bash
  python train.py --checkpoint_dir=<ckpt-path>

  Change path for speech_root, noise_root, and logdir_path in config.yaml
```

#### Inference

```bash
  python inference.py --checkpoint_dir=<ckpt-dir> --checkpoint_path=<checkpoint_file_name>
```
## Getting Models weights

| Model  | Description |  Link to the model | 
| :-------------: | :---------------: | :---------------: |
| speaker verfication | Weight for extracting speaker embedding |   [Link](https://drive.google.com/file/d/1ER2-zyMpDIpLYlCFl_skaiTTo2rQgqJw/view?usp=sharing) |
| Our model | Weights of the denoising model (needed for inference) | [Link](https://drive.google.com/file/d/1XNt8aKysxKTic1i4JZrgSvq3o0_UXrsh/view?usp=sharing)|
|[BigVGAN](https://github.com/NVIDIA/BigVGAN) | Weights of the fine tuned BigVGAN  | [Link](https://drive.google.com/drive/folders/1TNszyQfYMQzq0VM1RY2JJ3BVlWM9NTdC?usp=sharing)| 


## Citation
Please cite the following paper if you have used this code:
```bash
@inproceedings{Saha2024Approach,
  author = {Jayasree Saha and Rudrabha Mukhopadhyay and Aparna Agrawal and Surabhi Jain and C. V. Jawahar},
  title = {An Approach for Speech Enhancement in Low SNR Environments using Granular Speaker Embedding},
  booktitle = {7th Joint International Conference on Data Science \& Management of Data (11th ACM IKDD CODS and 29th COMAD) (CODS-COMAD 2024)},
  year = {2024},
  address = {Bangalore, India},
  publisher = {ACM},
  location = {New York, NY, USA},
  pages = {7},
  doi = {10.1145/3632410.3632413},
  url = {https://doi.org/10.1145/3632410.3632413}
}
```
