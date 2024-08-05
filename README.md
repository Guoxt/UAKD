### Uncertainty Driven Adaptive Self-Knowledge Distillation for Medical Image Segmentation
------
### Introduction 

<div style="text-align: justify;"> 

##### Deep learning have made great progress in medical image segmentation. However, the labels in the training set are often hard labels (i.e., one-hot vectors), which can easily lead to overfitting. To mitigate this problem, we propose an uncertainty driven adaptive self-knowledge distillation (UAKD) model for medical image segmentation, which regularizes the model training through the soft labels generated by itself. UAKD incorporates uncertainty estimation into the self-distillation framework, leverages teacher network ensembles to mitigate the semantic errors in the estimated soft labels caused by the fitting biases of the teacher networks. And we propose a novel adaptive distillation mechanism that leverages class uncertainty awareness to enhance the efficient transfer of knowledge from the teacher network to the student network. Further, we propose a cyclic ensemble method based on gradient ascent to estimate uncertainty. This approach improves the performance of UAKD compared to Monte Carlo dropout and significantly reduces computational costs compared to traditional deep ensemble methods.

</div>

------
### Framework

<img src="https://github.com/Guoxt/UAKD/blob/main/framework.png" alt="Image Alt Text" style="width:1000px; height:auto;">

------
### Run Code

1. train

```python main.py --patch_size 12 --in_channels 1 --T 2.5 --labels 2```                        # Setting Training Parameters

2. test

```python test.py --patch_size 12 --in_channels 1 --T 2.5 --labels 2```                        # Setting Testing Parameters

