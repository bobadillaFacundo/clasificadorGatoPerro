import numpy as np
from PIL import Image
import tensorflow as tf

# Carga el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="../assets/model.tflite")
interpreter.allocate_tensors()

# Obtén detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Crea una imagen de prueba (negra) del tamaño correcto
input_shape = input_details[0]['shape']
img = np.zeros((224, 224, 3), dtype=np.float32)  # Imagen negra normalizada [0,1]

# Si quieres probar con una imagen real:
# img = np.array(Image.open('ruta_a_una_foto.jpg').resize((224,224))).astype(np.float32) / 255.0

# Añade el batch dimension
input_data = np.expand_dims(img, axis=0)

# Pasa la imagen al modelo
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtén la predicción
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output_data)