# ChromaGAN-experience-reproduction
## Dataset

The dataset is taken from [the CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## Training:

- Execute Data.py first to generate `.npy` file for training:
```
python3 src/Data.py
```

- Execute train.py:
```
python3 src/train.py
```

- After that a file that saves model weights is saved at `saveModel` directory.

## Experimenting:
- To have colored images from test set run from root:
```
python3 src/coloringImages.py
```

## Resources

ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution. Patricia Vitoria, Lara Raad, Coloma Ballester. [link](https://openaccess.thecvf.com/content_WACV_2020/html/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.html)

U-Net: Convolutional Networks for Biomedical Image Segmentation. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. [link](https://arxiv.org/pdf/1505.04597)
