from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

def create_generators(train_dir, val_dir, test_dir, img_height=224, img_width=224, batch_size=8):
    # Stronger, still CPU-friendly augmentations for training.
    # These augmentations are intentionally conservative for medical images
    # (preserve lesion semantics while improving robustness).
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,            # random rotations
        width_shift_range=0.15,      # horizontal shifts
        height_shift_range=0.15,     # vertical shifts
        shear_range=0.15,            # shear transforms
        zoom_range=0.15,             # random zoom
        horizontal_flip=True,
        vertical_flip=True,          # some lesions appear in various orientations
        brightness_range=(0.7, 1.3), # brightness jitter
        channel_shift_range=20.0,    # small color perturbations
        fill_mode='reflect'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator
