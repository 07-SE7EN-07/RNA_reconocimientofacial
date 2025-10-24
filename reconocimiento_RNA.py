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
    Dense(3, activation='softmax')  # Cambié esto a 3 para las 3 personas que tienes
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generador de datos con aumento de datos (esto es útil cuando tienes pocas imágenes)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar el conjunto de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    'dataset/train',  # La ruta a la carpeta de entrenamiento
    target_size=(64, 64),
    batch_size=3,  # Usamos un batch pequeño ya que tienes pocas imágenes
    class_mode='sparse'  # Usamos sparse porque cada clase es un entero
)

# Cargar el conjunto de datos de prueba
test_generator = test_datagen.flow_from_directory(
    'dataset/test',  # La ruta a la carpeta de test
    target_size=(64, 64),
    batch_size=3,
    class_mode='sparse'
)

# Entrenar el modelo
model.fit(train_generator, epochs=10)

# Evaluar el modelo
model.evaluate(test_generator)
