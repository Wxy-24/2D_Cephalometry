# 2D_Cephalometry
------------------------------------------------------------------------------------------------------------------------------------------
Dataset is available from IEEE ISBI Challenge,you can download it from https://figshare.com/s/37ec464af8e81ae6ebbf

The task of this challenge is to localize the anatomical landmarks of skulls automatically.The evaluation metrics for this task is Euclidean distance between ground truth and prediction.

example of original image:
![image](http://github.com/Wxy-24/2D_Cephalometry/raw/master/2D_cephalometry/img/original.png)

example of landmark annotation:
![image](http://github.com/Wxy-24/2D_Cephalometry/raw/master/2D_cephalometry/img/annotation.png)

Here in this repo I compare the different method of localization including coordinate regression and prediction based on heatmap&argmax

I select 6 landmarks as targets and perform deep learning based method to localize them.Overall mean error of these landmark in a 4X downsampled images are shown as below(mean euclidean distance error is around 5.8 pixels)

sella turica:6.62

orbitale:4.83

porion:5.65

gnathion:3.89

gonion:8.82

anterior nasal spine:5.39

You can find code in the folder and Run *main.py* to see the perforance.
(You can also find examples in .ipynb format in another folder)
