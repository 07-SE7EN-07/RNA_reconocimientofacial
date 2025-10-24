import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir el modelo CNN para el reconocimiento facial
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Suponiendo que hay 10 clases de personas
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Cargar el conjunto de datos (esto puede variar dependiendo del dataset)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('ruta/a/tu/dataset', target_size=(64, 64), batch_size=32, class_mode='sparse')

# Entrenar el modelo
model.fit(train_generator, epochs=10)

# Evaluar el modelo (en un conjunto de datos de prueba)
test_generator = train_datagen.flow_from_directory('ruta/a/tu/test_dataset', target_size=(64, 64), batch_size=32, class_mode='sparse')
model.evaluate(test_generator)
