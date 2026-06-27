import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Parameters
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10

train_dir = r"C:\Users\Kanimozhi\Downloads\archive\Dataset\Train"
val_dir = r"C:\Users\Kanimozhi\Downloads\archive\Dataset\Val"

# Data Generators (Grayscale + Rescale)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',   # ✅ grayscale
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',   # ✅ grayscale
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# CNN Model (for grayscale → 1 channel)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save Model
model.save("drowsiness_model.h5")

print("✅ Model trained and saved successfully")
