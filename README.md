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
               [--transfer-learning] [--SVM-kernel SVM_KERNEL] [--reproduce]

P1 Cell Image Classification

optional arguments:
  -h, --help            show this help message and exit
  --space {raw,wndchrm,rcdt}
  --model {RF,KNN,SVM,LR,LDA,PLDA,MLP,ShallowCNN,VGG16,InceptionV3}
  --transfer-learning   neural network use pretrained weights instead of
                        training from scratch
  --SVM-kernel SVM_KERNEL
  --reproduce           reproduce the results reported in the paper
```

**Examples**

* Train A logistic regression model on image space: `python main.py --space raw --model LR`

* Train A logistic regression model on WND-CHARM feature space: `python main.py --space wndchrm --model LR`

* Train InceptionV3 on image space: `python main.py --space raw --model InceptionV3`

* Train InceptionV3 on image space by fine-tuning a pre-trained model (transfer learning): `python main.py --space image --model InceptionV3 --transfer-learning`

