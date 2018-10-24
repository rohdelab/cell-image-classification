
## Usage
        main.py [-h] {KNN,RF,LR,SVM,MLP,VGG16,DenseNet} {image,wndchrm}

## PERFORMANCE STATS
        VGG16 transfer learning (early stopping):
        Epoch 26/500
         - 8s - loss: 3.7375e-05 - acc: 1.0000 - val_loss: 0.9353 - val_acc: 0.8116
        689/689 [==============================] - 4s 5ms/step
        train accuracy: 0.9811320758177338
        173/173 [==============================] - 1s 8ms/step
        test accuracy: 0.8208092485549133
        
        Logistic Regression using wndchrm features:
        train accuracy: 1.0
        test accuracy: 0.9190751445086706
        
        DenseNet121 transfer learning (early stopping):
        Epoch 500/500
         - 8s - loss: 0.8985 - acc: 0.9984 - val_loss: 0.9542 - val_acc: 0.9710
        689/689 [==============================] - 3s 4ms/step
        train accuracy: 0.9956458635703919
        173/173 [==============================] - 1s 6ms/step
        test accuracy: 0.930635838150289
        
        
