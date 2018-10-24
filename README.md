
## Usage
    main.py [-h] {KNN,RF,LR,SVM,MLP,VGG16,DenseNet} {image,wndchrm}

## Dataset
    The 2D Hela Dataset (https://ome.irp.nia.nih.gov/iicbu2008/)
    Fluorescence microscopy images of HeLa cells of 10 classes
    860 382x382 16 bit TIFF images, of which 689 used for training and 173 for testing
    
## Neural Netwoks Training
    opt = keras.optimizers.RMSprop(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    early_stop = EarlyStopping(patience=20)
    model.fit(x_train, y_train, verbose=2, batch_size=32, epochs=500,
              validation_split=0.1, shuffle=True, callbacks=[early_stop])
## Performances
    VGG16 transfer learning (early stopping):
    Epoch 26/500
     - 8s - loss: 3.7375e-05 - acc: 1.0000 - val_loss: 0.9353 - val_acc: 0.8116
    train accuracy: 0.9811320758177338
    test accuracy: 0.8208092485549133

    DenseNet121 transfer learning (early stopping):
    Epoch 500/500
     - 8s - loss: 0.8985 - acc: 0.9984 - val_loss: 0.9542 - val_acc: 0.9710
    train accuracy: 0.9956458635703919
    test accuracy: 0.930635838150289

    Logistic Regression using wndchrm features:
    train accuracy: 1.0
    test accuracy: 0.9190751445086706
        

        
        
