# MRI-Generation
_Author: Antoine DELPLACE_  
_Last update: 10/02/2020_

This repository corresponds to the source code used for the MRI generation part of my Master Thesis entitled "__Segmentation and Generation of Magnetic Resonance Images by Deep Neural Networks__".

## Method description
The aim of the project is to generate high resolution brain Magnetic Resonance Images (MRIs) from a random latent space (noise) thanks to Generative Adversarial Networks (GANs). Three architectures are tuned and compared : Deep Convolutional GAN (__DCGAN__), Super Resolution Residual GAN (__SRResGAN__) and Progressive GAN (__ProGAN__).

## Usage

### Dependencies
- Python 3.6.8
- Tensorflow 1.14
- Numpy 1.16.2
- Matplotlib 3.0.3
- Imageio 2.5.0
- Scikit-image 0.15.0 -- `analyze_plots_prog.py`

### File description
1. The training files correspond to `dcgan3.py`, `srresgan1.py` and `progan5.py` in their respective folder.

2. The files `analyze_plots*.py` and `graph_loss*.py` enable the monitoring of the training process thanks to plots of the ouputs, the loss functions and the processing time. Note that you need to use the files ending by "_prog.py" for ProGAN.

3. The files `*_generate_data.py` and `*_generate_interpolation.py` enable the generation of images from the trained models.

4. The file `evaluate_gan.py` enables the evaluation of the performance of the model, based on the generated images.

## Results
The comparison and analysis of the different architectures can be found in the conference and thesis papers. Here, I just display the result of the training process, especially the __animations__ composed of the generated outputs during training.

### DCGAN
```sh
analyze_plots.py dcgan3
```
[training_animations/dcgan3.gif](http://antoine.delplace.eu/files/training_animations/dcgan3.gif)

### SRResGAN
```sh
analyze_plots.py srresgan1
```
[training_animations/srresgan1.gif](http://antoine.delplace.eu/files/training_animations/srresgan1.gif)

### ProGAN
```sh
analyze_plots_prog.py progan5
```
[training_animations/progan5.gif](http://antoine.delplace.eu/files/training_animations/progan5.gif)

## References
1. A. Delplace. "Synthetic Magnetic Resonance Images with Generative Adversarial Networks", _Conference paper at the University of Queensland_, October 2019. [arXiv:2002.02527](https://arxiv.org/abs/2002.02527)  
2. A. Delplace. "Segmentation and Generation of Magnetic Resonance Images by Deep Neural Networks", _Master thesis at the University of Queensland_, October 2019. [arXiv:2001.05447](https://arxiv.org/abs/2001.05447)