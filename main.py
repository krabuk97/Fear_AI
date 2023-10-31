import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Wczytanie gotowego modelu do rozpoznawania emocji
model = tf.keras.models.load_model('emotion_model.h5')

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)  # Uruchomienie kamery (można zmienić numer urządzenia)

while True:
    ret, frame = cap.read()  # Odczytanie ramki z kamery

    # Przetworzenie ramki (np. konwersja do skali szarości i zmiana rozmiaru)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))

    # Przygotowanie obrazu do predykcji przez model
    image = img_to_array(resized)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image / 255.0  # Normalizacja

    # Predykcja emocji
    predictions = model.predict(image)
    emotion_label = np.argmax(predictions)

    # Wyświetlenie ramki wideo z etykietą emocji
    cv2.putText(frame, "Emotion: " + str(emotion_label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wcisnij 'q' aby wyjść z pętli
        break

# Zamknięcie kamery i okien
cap.release()
cv2.destroyAllWindows()
