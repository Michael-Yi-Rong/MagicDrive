# MagicDrive

✨ Check out our new work [MagicDrive3D](https://github.com/flymin/MagicDrive3D) on **3D scene generation**!

✨ If you want **video generation**, please find the code at the [`video branch`](https://github.com/cure-lab/MagicDrive/tree/video).

[![arXiv](https://img.shields.io/badge/ArXiv-2310.02601-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2310.02601) [![web](https://img.shields.io/badge/Web-MagicDrive-blue.svg?style=plastic)](https://gaoruiyuan.com/magicdrive/) [![license](https://img.shields.io/github/license/cure-lab/MagicDrive?style=plastic)](https://github.com/cure-lab/MagicDrive/blob/main/LICENSE) [![star](https://img.shields.io/github/stars/cure-lab/MagicDrive)](https://github.com/cure-lab/MagicDrive)


# Cite Us

```bibtex
@inproceedings{gao2023magicdrive,
  title={{MagicDrive}: Street View Generation with Diverse 3D Geometry Control},
  author={Gao, Ruiyuan and Chen, Kai and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Yeung, Dit-Yan and Xu, Qiang},
  booktitle = {International Conference on Learning Representations},
  year={2024}
}
```

# Credit

We adopt the following open-sourced projects:

- [bevfusion](https://github.com/mit-han-lab/bevfusion): dataloader to handle 3d bounding boxes and BEV map
- [diffusers](https://github.com/huggingface/diffusers): framework to train stable diffusion
- [xformers](https://github.com/facebookresearch/xformers): accelerator for attention mechanism
- Thanks [@pixeli99](https://github.com/pixeli99) for training the [60-frame video generation](https://gaoruiyuan.com/magicdrive/#long-video).

# Demo

Original box & image
![0_ori_box](https://github.com/user-attachments/assets/b75eefe0-bda1-4a15-9391-7ac3dec1f2e9)
![0_ori](https://github.com/user-attachments/assets/518b3cdf-bb83-44b6-83eb-3c481ca8ed7b)

Generated box & image
![0_gen3_box](https://github.com/user-attachments/assets/ad2c3135-38b5-4c20-abb8-99518567b0a2)
![0_gen3](https://github.com/user-attachments/assets/87c6de32-7186-4505-a51f-f2f78898643a)
![0_gen2_box](https://github.com/user-attachments/assets/fa3e69e9-8056-4819-a213-98bac4676236)
![0_gen2](https://github.com/user-attachments/assets/027075bf-c194-44ff-80d3-4ae25ef5fc1a)
![0_gen1_box](https://github.com/user-attachments/assets/d0c09de6-295f-44d8-882c-dab551c9483f)
![0_gen1](https://github.com/user-attachments/assets/64da239a-dbd0-4ded-b7c0-ae1059c08c38)
![0_gen0_box](https://github.com/user-attachments/assets/42721ef1-9dc4-44a5-a893-32438e005f57)
![0_gen0](https://github.com/user-attachments/assets/f9c2f11c-46ab-4586-b58d-da1a6f673d46)
