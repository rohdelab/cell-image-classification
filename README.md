
Usage

    usage: main.py [-h] --space {raw,wndchrm,rcdt} --model
                   {PLDA,KNN,RF,LR,SVM,MLP,ShallowCNN,VGG16,InceptionV3,DenseNet}
                   [--transfer-learning] [--PCA-comps PCA_COMPS]
                   [--PLDA-comps PLDA_COMPS] [--PLDA-alpha PLDA_ALPHA]
                   [--SVM-kernel SVM_KERNEL] [--CV]
    
    P1 Cell Image Classification
    
    optional arguments:
      -h, --help            show this help message and exit
      --space {raw,wndchrm,rcdt}
      --model {PLDA,KNN,RF,LR,SVM,MLP,ShallowCNN,VGG16,InceptionV3,DenseNet}
      --transfer-learning   neural network use pretrained weights instead of
                            training from scratch
      --PCA-comps PCA_COMPS
                            number of PCA components if use PCA
      --PLDA-comps PLDA_COMPS
                            number of components if use PLDA
      --PLDA-alpha PLDA_ALPHA
                            PLDA alpha
      --SVM-kernel SVM_KERNEL
      --CV                  perform cross validation

Dataset

    The 2D Hela Dataset (https://ome.irp.nia.nih.gov/iicbu2008/)
    Fluorescence microscopy images of HeLa cells of 10 classes
    860 382x382 16 bit TIFF images, of which 689 used for training and 173 for testing
    
Neural Netwoks Training

    opt = keras.optimizers.RMSprop(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    early_stop = EarlyStopping(patience=20)
    model.fit(x_train, y_train, verbose=2, batch_size=32, epochs=500,
              validation_split=0.1, shuffle=True, callbacks=[early_stop])
Performances

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
        

        
        
