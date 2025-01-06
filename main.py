import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import kagglehub

# Chemins des données
REAL_FACES_DIR = kagglehub.dataset_download("tunguz/70000-real-faces-1")
FAKE_FACES_DIR = kagglehub.dataset_download("tunguz/1-million-fake-faces")

# Charger les images et leurs labels
def load_images(directory, label):
    images = []
    labels = []
    for root, _, files in os.walk(directory):  # Traverse directories recursively
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image extensions
                filepath = os.path.join(root, filename)
                try:
                    image = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
                    image = tf.keras.preprocessing.image.img_to_array(image)
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {filepath}: {e}")
    return images, labels

# Charger les données
real_images, real_labels = load_images(REAL_FACES_DIR, 0)
fake_images, fake_labels = load_images(FAKE_FACES_DIR, 1)

# Combiner et normaliser
images = np.array(real_images + fake_images, dtype="float32") / 255.0
labels = np.array(real_labels + fake_labels)

# Diviser les données
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Augmentation des données
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
train_generator = datagen.flow(X_train, y_train, batch_size=32)

# Construire le modèle
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Définir les callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Entraîner le modèle
history = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

# Évaluation et sauvegarde
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
model.save("face_classifier_model.h5")