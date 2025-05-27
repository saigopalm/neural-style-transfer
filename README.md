# Image Colorization Using Neural Style Transfer With CNNs

This project implements the technique described in the CVPR 2016 paper ["Image Style Transfer Using Convolutional Neural Networks"](https://arxiv.org/abs/1508.06576) by Gatys et al., adapted for artistic image colorization. The goal is to generate a color image given a grayscale image. Additionally, it can also be used to produce visually compelling outputs by blending the content of one image with the style of another, using deep convolutional neural networks.

The system extracts multi-level content and style features using a pre-trained **VGG-19** network and iteratively optimizes a target image to match both sets of features, effectively “colorizing” the content image with the style.


## Deployment

This project is deployed as a REST API using FastAPI.

Users can upload a content image and a style image and receive the stylized result via a POST request.
