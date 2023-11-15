# MSDR-Net-Multi-Scale-Detail-Recovery-Network-for-Single-Image-Deraining
MSDR-Net：Multi-Scale Detail-Recovery Network for Single Image Deraining
# Authors: Shuyu Han, Jun Wang*, Zaiyu Pan, Zhengwen Shen

# Abstract

Rain streaks vary in size, direction, and density, resulting in serious blurring and image quality degradation, which often directly affect the downstream visual tasks. At present, many end-to-end image removal networks have achieved good results, but image details are often lost during processing. Therefore, we propose a novel detail-recovery network to solve this problem. Unlike the existing works, we regard image rain removal and detail restoration as two different tasks simultaneously. Specifically, we use two encoder-decoder networks to extract rain streaks and detailed features and design different feature extraction blocks for two encoder-decoder networks. Due to the different receptive fields of feature layers at different scales, the information extracted at each scale is also different. The tasks of image rain removal and detail restoration are considered from the multi-scale feature level. To better respond to the image details and take full advantage of semantic information of multi-scale features, rain removal and image detail restoration are carried out at different scales. The proposed method has been validated on datasets to verify its effectiveness.


![image](https://github.com/B201cv/MSDR-Net-Multi-Scale-Detail-Recovery-Network-for-Single-Image-Deraining/assets/150791781/2a0c8fb8-d59d-4d70-9337-9a7166c59d12)

![image](https://github.com/B201cv/MSDR-Net-Multi-Scale-Detail-Recovery-Network-for-Single-Image-Deraining/assets/150791781/e6f5ee30-ce02-453c-b1f6-52dbdb0f1763)


# Results

![image](https://github.com/B201cv/MSDR-Net-Multi-Scale-Detail-Recovery-Network-for-Single-Image-Deraining/assets/150791781/81dfb861-49de-415a-bf38-c22f64a25b28)
![image](https://github.com/B201cv/MSDR-Net-Multi-Scale-Detail-Recovery-Network-for-Single-Image-Deraining/assets/150791781/50be0423-dda2-4612-8009-aef7f8c79bce)

# Citation
If you use the dataset, please cite the following paper:  
@inproceedings{han2022msdr,  
  title={MSDR-Net: Multi-Scale Detail-Recovery Network for Single Image Deraining},  
  author={Han, Shuyu and Wang, Jun and Pan, Zaiyu and Shen, Zhengwen},  
  booktitle={2022 China Automation Congress (CAC)},  
  pages={4823--4828},  
  year={2022},  
  organization={IEEE}  
}  
