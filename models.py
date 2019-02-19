import sys
sys.path.append('optimaltransport')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from optimaltransport.optrans.decomposition import PLDA
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline


def build_model(model_name, input_shape, num_classes, transfer_learning):
    from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
    from keras.models import Sequential
    from keras.applications import densenet
    from keras.applications import vgg16
    from keras.applications import InceptionV3
    from keras.models import Model

    weights = 'imagenet' if transfer_learning else None
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

    return model


def nn_clf(model_name, dataset, args):
    import keras
    from keras.callbacks import EarlyStopping
    x, y = dataset['x'], dataset['y']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y)
    input_shape = x_train.shape[1:]
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if model_name != 'MLP':
        x_train = np.reshape(x_train, [-1, *input_shape])
        x_test = np.reshape(x_test, [-1, *input_shape])
        if x_train.ndim == 3:
            x_train = np.repeat(x_train[..., np.newaxis], 3, axis=3)
            x_test = np.repeat(x_test[..., np.newaxis], 3, axis=3)
        else:
            assert x_train.ndim == 4
            assert x_train.shape[-1] == 3

    classnames = dataset['classnames']

    y_train = keras.utils.to_categorical(y_train, num_classes=len(classnames))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(classnames))

    if args.transfer_learning and model_name in ['VGG16', 'InceptionV3']:
        print("using pretrained weights {}".format(model_name))
    else:
        print("training {} from scratch...".format(model_name))

    model = build_model(model_name, x_train.shape[1:], len(classnames), args.transfer_learning)

    opt = keras.optimizers.Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    model.fit(x_train, y_train, verbose=2, batch_size=32, epochs=100,
              validation_split=0.1, shuffle=True, callbacks=[early_stop])

    _, train_acc = model.evaluate(x_train, y_train)
    _, test_acc = model.evaluate(x_test, y_test)
    print("train accuracy: {}, test accuracy {}".format(train_acc, test_acc))


def sklearn_clf(model_name, dataset, args):
    clf = {
        'RF': RandomForestClassifier(),
        'LR': LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced', multi_class='ovr'),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(kernel=args.SVM_kernel, class_weight='balanced', probability=True, decision_function_shape='ovr'),
        'PLDA': PLDA(),
        'LDA': PLDA(alpha=0.001)
    }[model_name]

    param_grid = {
        'RF': {}, 'KNN': {}, 'PLDA': {}, 'LDA': {},
        'LR': {'logisticregression__C': np.logspace(-4, 4, 9)},
        'SVM': {'svc__C': np.logspace(-2, 6, 9), 'svc__gamma': np.logspace(-4, 3, 8)}
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
