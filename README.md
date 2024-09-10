# Bag of Tricks
Investigating the Effects of Bag of Tricks for Object Classification Models: ResNet and ViT

## Abstract
Diverse methods exist for enhancing the performance of deep neural networks. These include various data pre- processing techniques and training strategies, collectively referred to as a “bag of tricks”. The objective of this report is to identify optimal combinations of these tricks that can yield superior performance. These tricks were applied in combinations to two object classification mod- els: ResNet and Vision Transformer (ViT). When trained using the CIFAR-10 dataset with appropriate combinations of tricks, I observed performance improvements of 9.77% and 10.14% for ResNet-50 and ViT-B/4, respectively, com- pared to their baseline versions. Thus, this empirical study underscores the importance of using different combinations of tricks in developing deep neural networks, highlighting their impact on model efficacy.

## View the full report
The report can be viwed by downloading the pdf file from my drive.  
The name of the pdf file is "Investigating the Effects of Bag of Tricks for Object Classification Models.pdf".  
https://drive.google.com/file/d/1pwk7WD8PPlMTbmxKyiC544wQfGuhoO-7/view?usp=sharing

## How to use the code
Follow the simple steps to test it on your own!
- clone the repository
- make sure the datasets are downloaded in the right folder
- you must have tensorboard installed. if not, delete the codes that are related to tensorboard
- now that you are ready, just do "python resnet.py" or python vit.py" to run the code!
- change the type of tricks to achieve the best performance!

## Results
### ResNet-50 result
![image](https://github.com/parkie0517/Bag_of_Tricks/assets/80407632/ef1af91c-e12b-4214-8264-8a9f2ad5aab2)

### ViT-B/4 result
![image](https://github.com/parkie0517/Bag_of_Tricks/assets/80407632/6dddc30a-b941-4585-9c8d-173c2c22ecb8)

## Conclusion and future work
In this report, I examined how various combinations of “bag of tricks” enhance the performance in training ResNet- 50 and ViT-B/4. It was observed that stacking most of these tricks generally improves performance. Techniques like geometric transformation, data normalization, and mixup, which belong to the data preprocessing category, have typ- ically been confirmed to boost model performance. How- ever, we also noted that tricks aimed at preventing overfit- ting, such as regularization techniques, are only beneficial when used appropriately. Therefore, this report emphasizes the importance of using an appropriate combination of bag of tricks when training deep neural networks, as determined through empirical studies.  
In the future, the focus will be on conducting the same experiment multiple times to enhance the reliability of the experiments. Additionally, there will be efforts to improve the performance of the models by applying regularization techniques in different ways, e.g. applying dropout at the Convolutional Layers. Finally, I am interested in verifying whether backbones like ResNet-50 or ViT-B/4 trained using the bag of tricks, positively impact the performance in more practical computer vision tasks such as object detection or semantic segmentation.

## Reference
- Implementation of ResNet-50 using PyTorch was done reffering to this link(https://github.com/AhnYoungBin/Resnet50_pytorch/blob/master/resnet_torch.ipynb)
- Implementation of ViT using PyTorch was done reffering to this link(https://github.com/parkie0517/Vision-Transformer-using-PyTorch-and-Tensorboard)
