import sys
sys.path.append('optimaltransport')
import numpy as np
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
        model = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        x = Flatten()(model.output)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=model.input, outputs=predictions)

    return base_model, model


def nn_clf(model_name, dataset, args):
    import tensorflow as tf
    x, y = dataset['x'], dataset['y']
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    acc = []
    for split, (train_idx, test_idx) in enumerate(cv.split(np.zeros(y.shape[0]), y)):
        print('training on split {}'.format(split))
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        input_shape = x_train.shape[1:]
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
        # scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)

        if model_name != 'MLP':
            x_train = np.reshape(x_train, [-1, *input_shape])
            x_test = np.reshape(x_test, [-1, *input_shape])
            if x_train.ndim == 3:
                x_train = np.repeat(x_train[..., np.newaxis], 3, axis=3)
                x_test = np.repeat(x_test[..., np.newaxis], 3, axis=3)
            else:
                assert x_train.ndim == 4
                assert x_train.shape[-1] == 3

        batch_size = 32
        if model_name == 'MLP':
            lr = 0.00001
        elif model_name == 'ShallowCNN':
            lr = 0.0005
        elif model_name == 'VGG16':
            lr = 0.00001
            batch_size = 16
        elif model_name == 'InceptionV3':
            lr = 0.00001
            batch_size = 8
        else:
            lr = 5e-4

        if 'classnames' in dataset:
            classnames = dataset['classnames']
        else:
            classnames = list(map(str, sorted(list(set(dataset['y'])))))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classnames))
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(classnames))

        if args.transfer_learning and model_name in ['VGG16', 'InceptionV3']:
            print("using pretrained weights {}".format(model_name))
        else:
            print("training {} from scratch...".format(model_name))

        base_model, model = build_model(model_name, x_train.shape[1:], len(classnames), args.transfer_learning)

        def optimize(lr):
            opt = tf.keras.optimizers.Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            early_stop = tf.keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
            model.fit(x_train, y_train, verbose=2, batch_size=batch_size, epochs=100,
                      validation_split=0.1, callbacks=[early_stop])

        if args.transfer_learning and base_model is not None:
            for layer in base_model.layers:
                layer.trainable = False
            optimize(lr)
            for layer in base_model.layers:
                layer.trainable = True
            optimize(lr/10.)
        else:
            optimize(lr)

        _, train_acc = model.evaluate(x_train, y_train)
        _, test_acc = model.evaluate(x_test, y_test)
        acc.append(test_acc)
        print("split {}, train accuracy: {}, test accuracy {}, running test accuracy {}".format(split, train_acc, test_acc, np.mean(acc)))
    print("10-fold cross validation accuracy: {:.4f} (+/- {:.4f})".format(np.mean(acc), np.std(acc)))


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
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    # scores = cross_val_score(pipeline_clf, x, y, cv=cv, n_jobs=-1)  # n_jobs=-1 to use all processors
    # print("10-fold cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
