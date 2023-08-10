import cv2

def preprocessing_algorithm(image):
  # Cargar la imagen del billete
  #image = cv2.imread(img_path, 1)

  # Convertir la imagen a escala de grises
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Normalizar la imagen en el rango [0, 255]
  normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)

  # Redimensionar la imágen
  res_image = cv2.resize(normalized_image, None, fx=0.4, fy=0.4, interpolation = cv2.INTER_AREA)

  # Aplicar la ecualización del histograma
  equalized_image = cv2.equalizeHist(res_image)

  # Crear un objeto CLAHE
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

  # Aplicar el CLAHE a la imagen de escala de grises
  enhanced_image = clahe.apply(equalized_image)

  return enhanced_image