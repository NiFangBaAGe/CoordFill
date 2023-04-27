
<h3> <a href='https://arxiv.org/abs/2303.08524'>[AAAI 2023 (Oral)] CoordFill: <br>
Efficient High-Resolution Image Inpainting via Parameterized Coordinate Querying </a> 
 </h3> 
<div>
    <a target='_blank'>Weihuang Liu <sup> 1,2</sup> </a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Xiaodong Cun <sup>*,2</sup></a>&emsp;
    <a href='https://www.cis.um.edu.mo/~cmpun/' target='_blank'>Chi-Man Pun <sup>*,1</sup></a>&emsp;
    <a href='https://menghanxia.github.io/' target='_blank'>Menghan Xia <sup>2</sup></a>&emsp;
    <a href='https://yzhang2016.github.io/' target='_blank'>Yong Zhang <sup>2</sup></a>&emsp; 
    <a href='https://juewang725.github.io/' target='_blank'>Jue Wang<sup>1</sup> </a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> University of Macau &emsp; <sup>2</sup> Tencent AI Lab &emsp; 
</div>
<br>


[comment]: <> (## 🚧 TODO)

[comment]: <> (- [X] Give detailed instruction.)

[comment]: <> (- [ ] Release inference code and checkpoints.)

[comment]: <> (- [ ] Release training code.)

## 🎼 Pipeline

![overview2](https://user-images.githubusercontent.com/4397546/225505967-f27e3649-6c25-4f61-a153-db4cfafbcbed.jpg)
Giving an input masked image, first, we resample the input image with a fixed resolution before fed to the network, which enables the contextual feature extraction under a unified receptive field. Then, we select the masked region features by the corresponding coordinates. For each spatial feature, we generate the required number of parameters by the linear mapping function, and the parameters are upsampled to the required resolution with nearest-neighbor interpolation. Next, we use the generated parameters to parameterize the pixel-wise querying network, Hence, we obtain a series of MLPs that are spatial-adaptive. Finally, the pixel-wise querying network takes the hole pixel’s coordinate as input and outputs the color values. 



## Environment
This code was implemented with Python 3.6 and PyTorch 1.8.1. You can install all the requirements via:
```bash
pip install -r requirements.txt
```

## Demo
```bash
python demo.py --input [INPUT_PATH] --mask [MASK_PATH] --config [CONFIG_PATH] --model [MODEL_PATH] --resolution [HEIGHT],[WIDTH]
```
`[INPUT_PATH]`: input image

`[MASK_PATH]`: input mask

`[CONFIG_PATH]`: config file

`[MODEL_PATH]`: backbone checkpoint

`[HEIGHT]`: target height

`[WIDTH]`: target width


## Train
Single GPU training: 
```bash
python train.py --config [CONFIG_PATH]
```
Multi GPU training:
```bash
python -m torch.distributed.launch --nproc_per_node=[NUM_GPU] train_parallel.py --config [CONFIG_PATH]
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Models

Please find the pre-trained models [here](https://uofmacau-my.sharepoint.com/:f:/g/personal/mc05379_umac_mo/Em6_auDrqwhKl34MO9w_AggBBMhI3lWb6pQfUbYqCFQ9ZA?e=xduixN).


## 🛎 Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{liu2023coordfill,
  title={CoordFill: Efficient High-Resolution Image Inpainting via Parameterized Coordinate Querying},
  author={Liu, Weihuang and Cun, Xiaodong and Pun, Chi-Man and Xia, Menghan and Zhang, Yong and Wang, Jue},
  booktitle={AAAI},
  year={2023}
}
```

## 💗 Acknowledgements

CoordFill code borrows heavily from [LIIF](https://github.com/yinboc/liif), [ASAPNet](https://github.com/tamarott/ASAPNet) and [LAMA](https://github.com/advimman/lama). We thank the author for sharing their wonderful code. 
