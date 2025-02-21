# EBDNet: Integrating Optical Flow with Kernel Prediction for Burst Denoising
Official TF2 implementation of "[EBDNet: Integrating Optical Flow with Kernel Prediction for Burst Denoising](https://ieeexplore.ieee.org/document/10487901/)", TCSVT 2024


# Requirements

- tensorflow==2.5.3
- numpy == 1.19.2
- opencv_python==4.5.4.60
- scikit_image==0.17.2

# Training

1. Download [Open Image Dataset](https://github.com/cvdfoundation/open-images-dataset).

2. Run `get_dataset_path.py` to get training paths for each image. 

3. For debug, run:
```
python train.py --debug
```

The original program runs on 4 GPUs. You can modify 'CUDA_VISIBLE_DEVICES' in the code according to your actual situation.

4. For grayscale burst denoising training, run:

```Python
python train.py
```

5. For color burst denoising training, run:

```Python
python train.py --color
```


# Testing

1. Download pre-trained models for grayscale and color burst denoising from [here](https://drive.google.com/drive/folders/1KMTR32Tx3V0FMu7wyv9gRU0P9uq15i7q?usp=sharing).

2. Download the grayscale test set provided by Mildenhall et al. from [here](https://drive.google.com/file/d/1UptBXV4f56wMDpS365ydhZkej6ABTFq1/view). Download the color test set supplied by Xia et al. from [here](https://drive.google.com/file/d/1rXmauXa_AW8ZrNiD2QPrbmxcIOfsiONE/view?usp=sharing)

3. For graycale testing with all gains, run:

```Python
python test_all_gain.py
```

4. For color testing with all gains, run:

```Python
python test_all_gain.py --color
```

# Acknowledgement

This code is based on [bpn](https://github.com/likesum/bpn) and [PWCNet-tf2](https://github.com/hellochick/PWCNet-tf2). Thanks for their awesome work.


# Citation

```bibtex
@article{pan2024ebdnet,
  title={EBDNet: Integrating Optical Flow With Kernel Prediction for Burst Denoising},
  author={Pan, Sicheng and Li, Yingming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
