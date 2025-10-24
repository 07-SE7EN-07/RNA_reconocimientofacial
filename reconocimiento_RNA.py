import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 clases: alvin, simon, teodoro
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(64, 64),
    batch_size=3,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(64, 64),
    batch_size=3,
    class_mode='sparse',
    shuffle=False  # importante para que las etiquetas coincidan
)

# Entrenamiento
history = model.fit(train_generator, validation_data=test_generator, epochs=50)

# Evaluación
model.evaluate(test_generator)

# Predicción
y_true = test_generator.classes
y_pred = model.predict(test_generator)
salida = np.argmax(y_pred, axis=1)

# Mostrar resultados
print("ETIQUETAS REALES:", y_true)
print("PREDICCIONES:", salida)

# Mostrar nombre de clases
class_names = list(train_generator.class_indices.keys())
print("NOMBRES DE CLASES:", class_names)

# Gráfico de precisión
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.grid()
plt.show()

# Mostrar una imagen del test
X_batch, y_batch = test_generator[0]
plt.imshow(X_batch[2])
plt.title(f"Clase real: {y_batch[2]}, Predicción: {salida[2]}")
plt.axis('off')
plt.show()