import cv2
import mediapipe as mp
import csv
import os

# --- Configurações Iniciais ---
# Inicializa o MediaPipe Hands
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(
    max_num_hands=1,  # Configurado para detectar apenas uma mão
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Nome do arquivo CSV para salvar os dados
nome_arquivo_csv = 'alfabeto_libras.csv'

# Número de amostras a serem coletadas por letra
num_amostras = 200

# --- Preparação do Arquivo CSV ---
# Cria o cabeçalho para o CSV
cabecalho = ['letra']
for i in range(21):
    cabecalho += [f'x{i}', f'y{i}', f'z{i}']

# Verifica se o arquivo já existe. Se não, cria e escreve o cabeçalho.
if not os.path.exists(nome_arquivo_csv):
    with open(nome_arquivo_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(cabecalho)

# --- Captura de Vídeo e Coleta ---
cap = cv2.VideoCapture(0)  # Inicia a webcam (pode ser 0 ou 1 dependendo do seu sistema)

while cap.isOpened():
    sucesso, imagem = cap.read()
    if not sucesso:
        print("Não foi possível acessar a câmera.")
        break

    # Inverte a imagem horizontalmente para um efeito de espelho
    imagem = cv2.flip(imagem, 1)

    # Converte a imagem de BGR para RGB (MediaPipe usa RGB)
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # Processa a imagem para detectar as mãos
    resultados = maos.process(imagem_rgb)

    # Desenha os pontos na mão se detectada
    if resultados.multi_hand_landmarks:
        for pontos_mao in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(
                imagem,
                pontos_mao,
                mp_maos.HAND_CONNECTIONS,
                mp_desenho.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),  # Pontos em azul
                mp_desenho.DrawingSpec(color=(0, 255, 0), thickness=2)  # Conexões em verde
            )

    # Exibe a imagem
    cv2.imshow('Coleta de Dados - Alfabeto em Libras', imagem)

    # Aguarda uma tecla ser pressionada
    tecla = cv2.waitKey(10) & 0xFF

    # Se a tecla 'q' for pressionada, sai do loop
    if tecla == ord('0'):
        break

    # Se a tecla for uma letra, inicia a coleta
    if ord('a') <= tecla <= ord('z'):
        letra_pressionada = chr(tecla)
        print(f"--- Coletando dados para a letra '{letra_pressionada.upper()}' ---")
        print("Mantenha a posição...")

        # Loop para coletar o número definido de amostras
        for i in range(num_amostras):
            # Recaptura a imagem para ter dados "frescos" em cada amostra
            sucesso_coleta, imagem_coleta = cap.read()
            if not sucesso_coleta:
                continue

            imagem_coleta = cv2.flip(imagem_coleta, 1)
            imagem_coleta_rgb = cv2.cvtColor(imagem_coleta, cv2.COLOR_BGR2RGB)
            resultados_coleta = maos.process(imagem_coleta_rgb)

            if resultados_coleta.multi_hand_landmarks:
                for pontos_mao_coleta in resultados_coleta.multi_hand_landmarks:
                    # Extrai as coordenadas e as achata em uma única lista
                    pontos = [letra_pressionada]
                    for marco in pontos_mao_coleta.landmark:
                        pontos.extend([marco.x, marco.y, marco.z])

                    # Salva a linha no arquivo CSV
                    with open(nome_arquivo_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(pontos)

            # Mostra um feedback visual durante a coleta
            texto_coletando = f"Coletando... {i + 1}/{num_amostras}"
            cv2.putText(imagem, texto_coletando, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Coleta de Dados - Alfabeto em Libras', imagem)
            cv2.waitKey(1)  # Essencial para a janela do OpenCV atualizar

        print(f"--- Coleta para '{letra_pressionada.upper()}' finalizada! ---")

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()