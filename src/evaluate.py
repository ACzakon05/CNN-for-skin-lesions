from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json
import csv
import os
import tensorflow.keras.backend as K
import tensorflow as tf

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:
    raise ImportError('scikit-learn is required for evaluation (classification_report, confusion_matrix).')

# Config
TEST_DIR = 'data/test'
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
MODEL_PATH = 'efficientnetb0_multi_class_cpu.h5'

# Create a non-shuffling test generator to preserve filename -> label order
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Focal loss (same as training) ---
def categorical_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = K.pow(1 - y_pred, gamma)
        if alpha is not None:
            alpha_tensor = K.constant(alpha, dtype=K.floatx())
            alpha_factor = y_true * alpha_tensor
            loss = alpha_factor * weight * cross_entropy
        else:
            loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return loss

# Load model without compiling (saved model references a custom loss)
model = load_model(MODEL_PATH, compile=False)

# Build alpha vector (upweight Melanoma if present) and recompile
num_classes = len(test_generator.class_indices)
alpha_vec = [1.0] * num_classes
mel_idx = test_generator.class_indices.get('Melanoma') if 'Melanoma' in test_generator.class_indices else None
if mel_idx is not None:
    alpha_vec[mel_idx] = 2.0

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=categorical_focal_loss(gamma=2.0, alpha=alpha_vec),
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Predict on entire test set
preds = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# Mapping indices -> class names
class_indices = test_generator.class_indices
inv_map = {v: k for k, v in class_indices.items()}
target_names = [inv_map[i] for i in range(len(inv_map))]

# Print classification report and confusion matrix
print('\nClassification report:')
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

print('Confusion matrix:')
print(confusion_matrix(y_true, y_pred))

# Save per-file predictions to CSV and JSON
out_csv = 'predictions.csv'
out_json = 'predictions.json'
filenames = test_generator.filenames

rows = []
for i, fname in enumerate(filenames):
    true_lbl = inv_map[y_true[i]]
    pred_lbl = inv_map[int(y_pred[i])]
    prob = float(np.max(preds[i]))
    full_probs = preds[i].tolist()  # full probability vector
    row = {'filename': fname, 'true_label': true_lbl, 'pred_label': pred_lbl, 'pred_prob': prob, 'full_probs': full_probs}
    rows.append(row)

with open(out_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'true_label', 'pred_label', 'pred_prob'])
    writer.writeheader()
    for r in rows:
        writer.writerow({k: v for k, v in r.items() if k != 'full_probs'})  # exclude full_probs for CSV

with open(out_json, 'w') as jf:
    json.dump(rows, jf, indent=2)

print(f"Saved predictions to {out_csv} and {out_json}")
