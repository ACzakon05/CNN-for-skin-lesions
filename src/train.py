from src.data_loader import create_generators
from src.model import create_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import json
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import random
import math
import tensorflow.keras.backend as K

# Parametry
batch_size = 8
epochs_head = 10
epochs_fine = 0 # zakomentowane w CPU-friendly

# Generatory danych
train_generator, val_generator, test_generator = create_generators(
    train_dir='data/train',
    val_dir='data/validate',
    test_dir='data/test',
    batch_size=batch_size
)


# --- Oversampled Sequence for training (balances classes by upsampling minority class) ---
class OversampledSequence(Sequence):
    def __init__(self, data_dir, class_indices, batch_size=8, img_size=(224,224), preprocess_fn=preprocess_input, shuffle=True):
        self.data_dir = data_dir
        self.class_indices = class_indices
        self.batch_size = batch_size
        self.img_size = img_size
        self.preprocess_fn = preprocess_fn
        self.shuffle = shuffle

        # gather files per class
        samples_per_class = {}
        for cls_name in class_indices.keys():
            cls_path = os.path.join(data_dir, cls_name)
            files = []
            if os.path.isdir(cls_path):
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        files.append(os.path.join(cls_path, fname))
            samples_per_class[cls_name] = files

        # compute max class count and oversample with replacement
        counts = {k: len(v) for k,v in samples_per_class.items()}
        max_count = max(counts.values()) if counts else 0
        self.samples = []
        for cls_name, files in samples_per_class.items():
            if not files:
                continue
            needed = max_count - len(files)
            # include existing files
            for f in files:
                self.samples.append((f, class_indices[cls_name]))
            # oversample randomly with replacement
            if needed > 0:
                for _ in range(needed):
                    f = random.choice(files)
                    self.samples.append((f, class_indices[cls_name]))

        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_samples), self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_samples), len(self.class_indices)), dtype=np.float32)
        for i, (fpath, cls_idx) in enumerate(batch_samples):
            img = image.load_img(fpath, target_size=self.img_size)
            arr = image.img_to_array(img)
            arr = self.preprocess_fn(arr)
            batch_x[i] = arr
            batch_y[i, cls_idx] = 1.0
        return batch_x, batch_y

# --- Focal loss implementation (multi-class) ---
def categorical_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = K.pow(1 - y_pred, gamma)
        if alpha is not None:
            # alpha can be list/array of shape (num_classes,)
            alpha_tensor = K.constant(alpha, dtype=K.floatx())
            alpha_factor = y_true * alpha_tensor
            loss = alpha_factor * weight * cross_entropy
        else:
            loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return loss

# create oversampled sequence for training
oversample = True
if oversample:
    # class_indices from train_generator maps class name -> index
    train_seq = OversampledSequence(data_dir='data/train', class_indices=train_generator.class_indices, batch_size=batch_size, img_size=(224,224))
else:
    train_seq = train_generator


with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# Tworzenie modelu z zamrożonym base_model (CPU-friendly)
model = create_model(num_classes=3, train_base=False)
# build alpha vector to upweight melanoma class in focal loss
num_classes = len(train_generator.class_indices)
alpha_vec = [1.0] * num_classes
mel_idx = train_generator.class_indices.get('Melanoma') if 'Melanoma' in train_generator.class_indices else None
if mel_idx is not None:
    # increase alpha for melanoma to emphasize it in focal loss
    alpha_vec[mel_idx] = 1.2

model.compile(optimizer='adam', loss=categorical_focal_loss(gamma=1.5, alpha=alpha_vec), metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
# Reduce LR on plateau for fine-tuning stability
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Trening tylko głowy klasyfikacyjnej
# compute simple class weights to mitigate imbalance (inverse-frequency)
num_classes = len(train_generator.class_indices)
counts = np.bincount(train_generator.classes, minlength=num_classes)
class_weights = {i: float(np.max(counts) / counts[i]) if counts[i] > 0 else 1.0 for i in range(num_classes)}

history = model.fit(
    train_seq,
    epochs=epochs_head,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights
)

# save head training history to disk for reproducibility
with open(f"history_head_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as hf:
    json.dump(history.history, hf)

# -----------------------------
# Fine-tuning – odblokowanie całego EfficientNet
# UWAGA: dla CPU trening będzie bardzo wolny
"""
Fine-tuning – unfreeze (part of) the EfficientNet base and train with low LR.
We access the base via `model.base_model` (set in `create_model`) and freeze early layers.
"""
base_model = getattr(model, 'base_model', None)
if base_model is not None:
    # Unfreeze the base model but freeze most of the earlier layers to preserve pretrained features
    base_model.trainable = True
    # Freeze first N layers (adjust N depending on your compute / dataset size)
    freeze_until = -20  # keep last 20 layers trainable
    if freeze_until < 0:
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

    # recompile with a much lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=categorical_focal_loss(gamma=1.5, alpha=alpha_vec),
                  metrics=['accuracy'])

    history_fine = model.fit(
        train_seq,
        epochs=epochs_fine,
        validation_data=val_generator,
        callbacks=[early_stop, checkpoint, reduce_lr],
        class_weight=class_weights
    )

    # save fine-tuning history too
    with open(f"history_fine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as hf:
        json.dump(history_fine.history, hf)
# -----------------------------

# Wykres historii treningu
acc = history.history['accuracy']  # + history_fine.history['accuracy'] jeśli odblokujesz fine-tuning
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# if we ran fine tuning, append its history for plotting
if 'history_fine' in locals():
    acc += history_fine.history.get('accuracy', [])
    val_acc += history_fine.history.get('val_accuracy', [])
    loss += history_fine.history.get('loss', [])
    val_loss += history_fine.history.get('val_loss', [])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(acc, label='train accuracy')
plt.plot(val_acc, label='val accuracy')
plt.title('Dokładność')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.title('Strata')
plt.xlabel('Epoka')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Zapis modelu
model.save('efficientnetb0_multi_class_cpu.h5')
print('Model and histories saved. class_indices.json also written.')
