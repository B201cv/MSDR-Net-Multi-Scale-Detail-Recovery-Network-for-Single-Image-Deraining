# MSDR-Net-Multi-Scale-Detail-Recovery-Network-for-Single-Image-Deraining
MSDR-Net：Multi-Scale Detail-Recovery Network for Single Image Deraining

Abstract 
Rain streaks vary in size, direction, and density, resulting in serious blurring and image quality degradation, which often directly affect the downstream visual tasks. At present, many end-to-end image removal networks have achieved good results, but image details are often lost during processing. Therefore, we propose a novel detail-recovery network to solve this problem. Unlike the existing works, we regard image rain removal and detail restoration as two different tasks simultaneously. Specifically, we use two encoder-decoder networks to extract rain streaks and detailed features and design different feature extraction blocks for two encoder-decoder networks. Due to the different receptive fields of feature layers at different scales, the information extracted at each scale is also different. The tasks of image rain removal and detail restoration are considered from the multi-scale feature level. To better respond to the image details and take full advantage of semantic information of multiscale features, rain removal and image detail restoration are carried out at different scales. The proposed method has been validated on datasets to verify its effectiveness.


![image](https://github.com/B201cv/MSDR-Net-Multi-Scale-Detail-Recovery-Network-for-Single-Image-Deraining/assets/150791781/322dcda4-a3c2-40a3-91bd-4baa1c7dcd78)
