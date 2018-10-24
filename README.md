
Usage
        main.py [-h] {KNN,RF,LR,SVM,MLP,VGG16,DenseNet} {image,wndchrm}

PERFORMANCE STATS
        VGG16 transfer learning performance:
        Epoch 26/500
         - 8s - loss: 3.7375e-05 - acc: 1.0000 - val_loss: 0.9353 - val_acc: 0.8116
        689/689 [==============================] - 4s 5ms/step
        train accuracy: [0.09367595116229115, 0.9811320758177338]
        173/173 [==============================] - 1s 8ms/step
        test accuracy: [0.6480065168021516, 0.8208092485549133]
                      precision    recall  f1-score   support

        microtubules       0.83      0.79      0.81        19
              golgia       0.57      0.73      0.64        11
              golgpp       0.71      0.62      0.67        16
               actin       1.00      0.95      0.97        20
                  er       0.90      0.90      0.90        21
            endosome       0.65      0.74      0.69        23
            lysosome       0.87      0.62      0.72        21
           nucleolus       0.92      1.00      0.96        12
                 dna       1.00      1.00      1.00        13
        mitochondria       0.80      0.94      0.86        17

           micro avg       0.82      0.82      0.82       173
           macro avg       0.83      0.83      0.82       173
        weighted avg       0.83      0.82      0.82       173

        confusion matrix:
        [[15  0  0  0  2  0  0  0  0  2]
         [ 0  8  3  0  0  0  0  0  0  0]
         [ 0  4 10  0  0  0  1  1  0  0]
         [ 1  0  0 19  0  0  0  0  0  0]
         [ 0  0  0  0 19  1  0  0  0  1]
         [ 2  2  0  0  0 17  1  0  0  1]
         [ 0  0  0  0  0  8 13  0  0  0]
         [ 0  0  0  0  0  0  0 12  0  0]
         [ 0  0  0  0  0  0  0  0 13  0]
         [ 0  0  1  0  0  0  0  0  0 16]]
