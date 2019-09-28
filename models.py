import sys
sys.path.append('optimaltransport')
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from optimaltransport.optrans.decomposition import PLDA
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline


def build_model(model_name, input_shape, num_classes, transfer_learning):
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import densenet
    from tensorflow.keras.applications import vgg16
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model

    weights = 'imagenet' if transfer_learning else None
    base_model = None
    if model_name == 'MLP':
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
    elif model_name == 'ShallowCNN':
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=[11, 5], strides=[3, 3], input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=[3, 3], strides=[2, 2]))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
    elif model_name == 'VGG16':
        base_model = vgg16.VGG16(weights=weights, include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights=weights, include_top=False, input_shape=input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(4096, activation='relu')(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    elif model_name == 'DenseNet':
        base_model = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    elif model_name == 'ResNet':
        print('Building RestNet model...')
        base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model


def nn_clf(model_name, dataset, args):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    x, y = dataset['x'], dataset['y']
    if 'classnames' in dataset:
        classnames = dataset['classnames']
    else:
        classnames = list(map(str, sorted(list(set(y)))))

    """
    input_shape = x.shape[1:]
    scaler = StandardScaler()
    x = scaler.fit_transform(x.reshape([x.shape[0], -1]))
    x = x.reshape([x.shape[0], *input_shape])
    """

    if model_name == 'MLP':
        x = np.reshape(x, (x.shape[0], -1))
    else:
        if x.ndim == 3: # grayscale image
            x = np.repeat(x[..., np.newaxis], 3, axis=3)
        else:
            assert x.ndim == 4
            assert x.shape[-1] == 3

    batch_size = 32
    lr = {'MLP': 1e-5, 'ShallowCNN': 5e-4, 'VGG16': 1e-5, 'InceptionV3': 1e-5, 'ResNet': 1e-5, 'DenseNet': 1e-5}[model_name]
    batch_size = {'MLP': 32, 'ShallowCNN': 32, 'VGG16': 16, 'InceptionV3': 8, 'ResNet': 16, 'DenseNet': 16}[model_name]
    validation_split = 0.1

    if args.transfer_learning and model_name not in ['MLP', 'ShallowCNN']:
        print("using pretrained weights {}".format(model_name))
    else:
        print("training {} from scratch...".format(model_name))

    def optimize(lr):
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0001, patience=5, verbose=2, mode='auto')
        if args.data_augmentation:
          model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), verbose=2,
                              steps_per_epoch=len(x_train) / batch_size, epochs=100, 
                              validation_data=(x_train_val, y_train_val), callbacks=[early_stop])
        else:
          model.fit(x_train, y_train, verbose=2, batch_size=batch_size, epochs=100,
                    validation_split=validation_split, callbacks=[early_stop])

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y)
    cv = StratifiedKFold(n_splits=args.splits, shuffle=True)
    acc = []
    confs = []
    for split, (train_idx, test_idx) in enumerate(cv.split(np.zeros(y.shape[0]), y)):
        tf.keras.backend.clear_session() 
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        print('============ training on split {}, training samples {}, test samples {}'.format(split, x_train.shape[0], x_test.shape[0]))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classnames))
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(classnames))
        
        if args.data_augmentation:
          val_samples = int(x_train.shape[0]*validation_split)
          x_train, y_train = x_train[:-val_samples], y_train[:-val_samples]
          x_train_val, y_train_val = x_train[-val_samples:], y_train[-val_samples:]
          print('split training samples {}, validation samples {}'.format(x_train.shape[0], x_train_val.shape[0]))
          datagen = ImageDataGenerator(
              #featurewise_center=True,
              #featurewise_std_normalization=True,
              rotation_range=20,
              width_shift_range=0.2,
              height_shift_range=0.2,
              horizontal_flip=True)

          # # compute quantities required for featurewise normalization, not necessary here
          # datagen.fit(x_train)
  
        base_model, model = build_model(model_name, x.shape[1:], len(classnames), args.transfer_learning)

        if args.transfer_learning:
            for layer in base_model.layers:
                layer.trainable = False
            print('finetuning last layer...')
            optimize(lr)
            for layer in base_model.layers:
                layer.trainable = True
            print('finetunning the whole network...')
            optimize(lr/10.)
        else:
            optimize(lr)

        _, train_acc = model.evaluate(x_train, y_train, verbose=0)
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        acc.append(test_acc)
        y_pred = np.argmax(model.predict(x_test), 1)
        confs.append(confusion_matrix(np.argmax(y_test, 1), y_pred))
        print('running confusion matrix:')
        print(np.stack(confs, 0).mean(axis=0))
        print("split {}, train accuracy: {}, test accuracy {}, running test accuracy {}".format(split, train_acc, test_acc, np.mean(acc)))
        Path('checkpoints/{}/{}'.format(args.dataset, args.model)).mkdir(parents=True, exist_ok=True)
        model.save_weights('checkpoints/{}/{}/model_T{}U{}split{}.h5'.format(args.dataset, args.model, int(args.transfer_learning), int(args.data_augmentation), split))

    print("{}-fold cross validation accuracy: {:.4f} (+/- {:.4f})".format(args.splits, np.mean(acc), np.std(acc)))
    print('confusion matrix:')
    print(np.stack(confs, 0).mean(axis=0))


def sklearn_clf(model_name, dataset, args):
    if 'SVM' in model_name:
        model_name = 'SVM-L' if args.SVM_kernel == 'linear' else 'SVM-K'
    clf = {
        'RF': RandomForestClassifier(),
        'LR': LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced', multi_class='ovr'),
        'KNN': KNeighborsClassifier(),
        'SVM-L': LinearSVC(),
        'SVM-K': SVC(kernel=args.SVM_kernel, class_weight='balanced', probability=True, decision_function_shape='ovr'),
        'PLDA': PLDA(),
        'LDA': PLDA(alpha=0.001)
    }[model_name]

    param_grid = {
        'RF': {}, 'KNN': {}, 'PLDA': {}, 'LDA': {},
        'LR': {'logisticregression__C': np.logspace(-4, 4, 9)},
        'SVM-L': {},
        'SVM-K': {'svc__C': np.logspace(-2, 6, 9), 'svc__gamma': np.logspace(-4, 3, 8)}
    }[model_name]

    x, y = dataset['x'], dataset['y']
    x = np.reshape(x, (x.shape[0], -1))
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    pipeline_clf = make_pipeline(StandardScaler(), PCA(min(x.shape[0], x.shape[1])), clf)

    print("training ...")
    search = GridSearchCV(pipeline_clf, param_grid, cv=cv, n_jobs=-1)
    search.fit(x, y)

    estimator = search.best_estimator_

    confs = []
    for split, (train_idx, test_idx) in enumerate(cv.split(np.zeros(y.shape[0]), y)):
        x_test, y_test = x[test_idx], y[test_idx]
        y_pred = estimator.predict(x_test)
        confs.append(confusion_matrix(y_test, y_pred))

    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    print('confusion matrix:')
    print(np.stack(confs, 0).mean(axis=0))
    # scores = cross_val_score(pipeline_clf, x, y, cv=cv, n_jobs=-1)  # n_jobs=-1 to use all processors
    # print("10-fold cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
