# Contrast-enhanced MRI synthesis using dense-dilated residual convolutions based 3D network toward elimination of gadolinium in neuro-oncology
Recent studies have raised broad safety and health concerns about using of gadolinium contrast agents during magnetic resonance imaging (MRI) to enhance identification of active tumors. In this paper, we developed a deep learning-based method for three-dimensional (3D) contrast-enhanced T1-weighted (T1) image synthesis from contrast-free image(s).


![image](https://github.com/afiosman/dense-dilated-residual-convolutions-for-contrast-enhanced-MRI-synthesis/assets/10604649/32454618-4116-47da-835e-db899b3274ec)

Fig. 1. Our 3D DD-Res U-Net architecture for contrast-enhanced T1 MR image synthesis from contrast-free image(s). Each blue box represents a set of feature maps. The number on top of the box donates the extracted feature maps, and that at the left/right side of the box represents the size of the feature maps. White boxes represent copied feature maps. The arrows denote the different operations.


![image](https://github.com/afiosman/dense-dilated-residual-convolutions-for-contrast-enhanced-MRI-synthesis/assets/10604649/f1c340ee-895a-4e1d-b1a9-2f608ad1e1c7)

Fig. 2. Visual assessment of our proposed DD-Res U-Net model for contrast-enhanced T1 MRI synthesis and a 3D U-Net baseline model for 6 patients on the test set. From left to right: contrast-enhanced T1 synthetic image using only T1 as input (U-Net); contrast-enhanced T1 synthetic image using only T1 as input (the proposed model); contrast-enhanced T1 synthetic image using T1, T2, & FLAIR as input (the proposed model); ground-truth image; and residuals (absolute difference between synthetic and ground-truth). Rows show results for different patients.


# Availability of data and materials
The datasets generated during and/or analyzed during the current study are available at the RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2021 repository http://braintumorsegmentation.org/. 

# Paper
Please cite this paper: Osman AFI, Tamam NM. Contrast-enhanced MRI synthesis using dense-dilated residual convolutions based 3D network toward elimination of gadolinium in neuro-oncology. Journal of Applied Clinical Medical Physics Medical Physics 2023;1-8. https://doi.org/10.1002/acm2.14120 
