import os
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow.keras.backend as K
import tensorflow as tf

# ----------- CONFIG -----------
MODEL_PATH = "efficientnetb0_multi_class_cpu.h5"
CLASS_MAP_PATH = "class_indices.json"
IMG_SIZE = (224, 224)
# ------------------------------


def load_class_indices():
    """Wczytuje mapę klas """
    with open(CLASS_MAP_PATH, "r") as f:
        class_indices = json.load(f)

    # Odwrócenie mapy:
    inv_map = {v: k for k, v in class_indices.items()}
    return {v: k for k, v in class_indices.items()}, inv_map


def load_and_prepare_image(img_path):
    """Wczytuje obraz i przetwarza go dla EfficientNetB0"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_single_image(model, img_path, classes):
    """Predykcja dla jednego obrazu"""
    img_array = load_and_prepare_image(img_path)
    preds = model.predict(img_array)
    # return top-3 predictions as (class, prob) tuples
    probs = preds[0]
    top_k = min(3, probs.shape[0])
    top_idx = probs.argsort()[-top_k:][::-1]
    results = [(classes[int(i)], float(probs[int(i)])) for i in top_idx]
    return results



def is_image_file(filename):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    return filename.lower().endswith(exts)


# --- Focal loss (same as training) ---
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


if __name__ == "__main__":
    # Ładowanie klas
    classes, _ = load_class_indices()

    # Build alpha vector (upweight Melanoma if present) and load model
    num_classes = len(classes)
    alpha_vec = [1.0] * num_classes
    mel_idx = None
    for idx, cls_name in classes.items():
        if cls_name == 'Melanoma':
            mel_idx = idx
            break
    if mel_idx is not None:
        alpha_vec[mel_idx] = 1.2

    # Load model without compiling (saved model references a custom loss)
    print("Ładowanie modelu...")
    model = load_model(MODEL_PATH, compile=False)

    # Recompile with focal loss
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=categorical_focal_loss(gamma=1.5, alpha=alpha_vec),
                  metrics=['accuracy'])

    print("Model załadowany. Podaj ścieżkę do obrazu lub folderu:")
    path = input("> ").strip()

    if os.path.isfile(path):
        # Jeden obraz
        if not is_image_file(path):
            print("To nie wygląda na plik graficzny!")
        else:
            results = predict_single_image(model, path, classes)
            print("\n📌 Top predictions:")
            for cls, prob in results:
                print(f" - {cls}: {prob:.3f}")
    
    elif os.path.isdir(path):
        print("\n📁 Wykryto folder. Przetwarzam wszystkie obrazy...\n")
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if is_image_file(fname):
                results = predict_single_image(model, fpath, classes)
                top = results[0]
                print(f"{fname:<25} → {top[0]} ({top[1]:.2f})")
    else:
        print("❌ Nie znaleziono pliku ani folderu.")
