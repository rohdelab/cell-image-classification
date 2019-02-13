# Cell Image Classification

Code for training and testing of a set of statistical machine learning models on the 2D HeLa dataset (https://ome.irp.nia.nih.gov/iicbu2008/hela/index.html).

## Dependencies

* TensorFlow https://www.tensorflow.org/
* Keras https://keras.io/
* WND-CHARM Python API https://github.com/wnd-charm/wnd-charm

## Usage

```
usage: main.py [-h] --space {raw,wndchrm,rcdt} --model
               {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3}
               [--transfer-learning] [--PCA-comps PCA_COMPS]
               [--PLDA-comps PLDA_COMPS] [--PLDA-alpha PLDA_ALPHA]
               [--SVM-kernel SVM_KERNEL]

P1 Cell Image Classification

optional arguments:
  -h, --help            show this help message and exit
  --space {raw,wndchrm,rcdt}
  --model {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3}
  --transfer-learning   neural network use pretrained weights instead of
                        training from scratch
  --PCA-comps PCA_COMPS
                        number of PCA components if use PCA
  --PLDA-comps PLDA_COMPS
                        number of components if use PLDA
  --PLDA-alpha PLDA_ALPHA
                        PLDA alpha
  --SVM-kernel SVM_KERNEL
```

**Examples**

* Train A logistic regression model on image space: `python main.py --space raw --model LR`

* Train A logistic regression model on WND-CHARM feature space: `python main.py --space wndchrm --model LR`

* Train InceptionV3 on image space: `python main.py --space raw --model InceptionV3`

* Train InceptionV3 on image space by fine-tuning a pre-trained model (transfer learning): `python main.py --space image --model InceptionV3 --transfer-learning`

## Model Performances
```
VGG16 transfer learning:
Epoch 26/500
 - 8s - loss: 3.7375e-05 - acc: 1.0000 - val_loss: 0.9353 - val_acc: 0.8116
train accuracy: 0.9811320758177338
test accuracy: 0.8208092485549133

DenseNet121 transfer learning:
Epoch 500/500
 - 8s - loss: 0.8985 - acc: 0.9984 - val_loss: 0.9542 - val_acc: 0.9710
train accuracy: 0.9956458635703919
test accuracy: 0.930635838150289

Logistic Regression using wndchrm features:
train accuracy: 1.0
test accuracy: 0.9190751445086706
```