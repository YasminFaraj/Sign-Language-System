import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time

# Carrega o modelo e label encoder
model = load_model('modelo_libras.h5')
label_encoder = joblib.load('label_encoder.pkl')

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

nome_reconhecido = ""
ultima_letra = ""
tempo_ultima_letra = 0
intervalo_segundos = 1.0  # intervalo mínimo entre letras

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            dados = []
            for lm in landmarks.landmark:
                dados.extend([lm.x, lm.y, lm.z])

            if len(dados) == 63:
                entrada = np.array(dados).reshape(1, -1)
                predicao = model.predict(entrada)
                letra = label_encoder.inverse_transform([np.argmax(predicao)])[0]

                tempo_atual = time.time()
                if letra != ultima_letra and (tempo_atual - tempo_ultima_letra) > intervalo_segundos:
                    nome_reconhecido += letra
                    ultima_letra = letra
                    tempo_ultima_letra = tempo_atual

    # Mostrar nome na tela
    cv2.putText(frame, f"Nome: {nome_reconhecido}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cv2.putText(frame, "Aperte C para limpar, 0 para sair", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

    cv2.imshow("Reconhecimento LIBRAS", frame)

    tecla = cv2.waitKey(10) & 0xFF
    if tecla == ord('0'):
        break
    if tecla == ord('c'):
        nome_reconhecido = ""
        ultima_letra = ""
        tempo_ultima_letra = 0

cap.release()
cv2.destroyAllWindows()
