# EBDNet

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




# Acknowledgement

This code is based on [bpn](https://github.com/likesum/bpn) and [PWCNet-tf2](https://github.com/hellochick/PWCNet-tf2). Thanks for their awesome work.


# Citation
