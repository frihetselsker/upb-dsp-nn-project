import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from tqdm import tqdm
from python_speech_features import mfcc


# =========================
# Feature extraction
# =========================
def build_rand_feat(file_list, n, _min=None, _max=None):
    """
    _min/_max: if provided, use them for normalization (val set).
               if None, compute from this batch (train set) and return them.
    """
    X = []
    y = []
    local_min, local_max = float('inf'), -float('inf')

    file_list = np.array(file_list)

    for _ in tqdm(range(n)):
        file = np.random.choice(file_list)

        rate, wav = wavfile.read('clean/' + file)
        label = df.at[file, 'label']

        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index:rand_index + config.step]

        X_sample = mfcc(
            sample,
            rate,
            numcep=config.nfeat,
            nfilt=config.nfilt,
            nfft=config.nfft
        ).T

        local_min = min(local_min, np.min(X_sample))
        local_max = max(local_max, np.max(X_sample))

        X.append(X_sample if config.mode == 'conv' else X_sample.T)
        y.append(label_to_index[label])

    X, y = np.array(X), np.array(y)

    # FIX 1: Use train min/max for val normalization to avoid data leakage
    if _min is None or _max is None:
        _min, _max = local_min, local_max

    X = (X - _min) / (_max - _min + 1e-9)

    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    y = to_categorical(y, num_classes=len(class_names))

    return X, y, _min, _max


# =========================
# Model
# =========================
def get_conv_model(input_shape):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     strides=(1, 1), padding='same',
                     input_shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    # FIX 4: Use dynamic class count instead of hardcoded 10
    model.add(Dense(len(class_names), activation="softmax"))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    return model


# =========================
# Config
# =========================
class Config:
    def __init__(self, mode='conv', nfilt=40, nfeat=20, nfft=2048, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfft = nfft
        self.nfeat = nfeat
        self.rate = rate
        self.step = int(rate / 2)


# =========================
# Load data
# =========================
df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/' + f)
    df.at[f, 'length'] = signal.shape[0] / rate

class_names = list(np.unique(df.label))
label_to_index = {c: i for i, c in enumerate(class_names)}

class_dist = df.groupby(['label'])['length'].mean()

# FIX 2: Compute total samples based on full dataset; split proportionally later
n_samples_total = 2 * int(df['length'].sum() / 0.1)
prob_dist = class_dist / class_dist.sum()


# =========================
# Train/Val split (FILE LEVEL)
# =========================
train_files, val_files = train_test_split(
    df.index.unique(),
    test_size=0.2,
    random_state=42,
    stratify=df.label
)

# FIX 2: Scale sample counts proportionally to split sizes
n_train = int(n_samples_total * 0.8)
n_val   = int(n_samples_total * 0.2)


# =========================
# Build dataset
# =========================
config = Config(mode='conv')

X_train, y_train, train_min, train_max = build_rand_feat(train_files, n_train)
# FIX 1: Pass train min/max to val so normalization is consistent
X_val, y_val, _, _ = build_rand_feat(val_files, n_val, _min=train_min, _max=train_max)

y_train_flat = np.argmax(y_train, axis=1)


# =========================
# Model init
# =========================
input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = get_conv_model(input_shape)


# =========================
# Class weights
# =========================
# FIX 3: Compute class weights from actual training file labels, not random samples
train_labels = df.loc[train_files, 'label'].map(label_to_index).values
classes = np.unique(train_labels)

class_weights_raw = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_labels
)

class_weight = dict(zip(classes, class_weights_raw))


# =========================
# Training
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weight
)


# =========================
# Evaluation
# =========================
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap='Blues', xticks_rotation=45)
plt.show()

print(classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names
))