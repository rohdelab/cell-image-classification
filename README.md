# Cell Image Classification Toolbox

A toolbox for training cell images classification models. Supported models include neural networks (MLP, customized CNN, VGG, InceptionV3, ResNet, etc.), and statistical ML models (SVM, logitic regression, random forest, etc.). Train models with cross-validation, data augmentation and fine-tuning.

This repo accompanies the paper "Cell image classification: a comparative overview" (https://arxiv.org/abs/1906.03316)

## Dependencies

* TensorFlow 1.13.1 https://www.tensorflow.org/
* WND-CHARM, and its Python API https://github.com/wnd-charm/wnd-charm
* scikit-learn 0.18.1 <https://scikit-learn.org/stable/>
* Optimal transport library (https://github.com/LiamCattell/optimaltransport)

## Usage

```
usage: main.py [-h] [--dataset DATASET] --space {image,wndchrm,rcdt} --model
               {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3,ResNet,DenseNet}
               [-T] [-U] [--splits {2,3,4,5,6,7,8,9,10}]
               [--SVM-kernel {rbf,linear}] [--preprocessed]
               --target_image_size {32,64,75,128,256}

P1 Cell Image Classification

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET
  --space {image,wndchrm,rcdt}
  --model {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3,ResNet,DenseNet}
  -T, --transfer-learning
                        neural network use pretrained weights instead of
                        training from scratch
  -U, --data_augmentation
                        use data augmentation for neural network based
                        approaches
  --splits {2,3,4,5,6,7,8,9,10}
                        number of splits for cross-validation
  --SVM-kernel {rbf,linear}
  --preprocessed        reproduce the results on Hela dataset reported in the
                        paper
  --target_image_size {32,64,75,128,256}
                        image size used for classification
```

**Examples**

* Train A logistic regression model on image space: `python main.py --space image --model LR`

* Train A logistic regression model on WND-CHARM feature space: `python main.py --space wndchrm --model LR`

* Train InceptionV3 on image space: `python main.py --space image --model InceptionV3`

* Train InceptionV3 on image space by fine-tuning a pre-trained model (transfer learning): `python main.py --space image --model InceptionV3 --transfer-learning`

**Reproduce Hela Results**

We provide the data used for producing the Hela results as reported in the paper. The preprocessed data is  located in `data/hela_preprocessed`. To reproduce the results, add the `--preprocessed` flag to commands (this will load the propressed data). For example, to reproduce PLDA classification on wndchrm features result, run `python main.py --dataset hela --space wndchrm --model PLDA --preprocessed`.
