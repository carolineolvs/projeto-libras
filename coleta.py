import cv2
import mediapipe as mp
import csv
import os

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
# Blindagem ativada: max_num_hands=1
maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

camera = cv2.VideoCapture(0)
arquivo_csv = 'dados_libras.csv'

if not os.path.exists(arquivo_csv):
    with open(arquivo_csv, mode='w', newline='') as f:
        escritor = csv.writer(f)
        cabecalho = ['letra']
        for i in range(21):
            cabecalho.extend([f'x{i}', f'y{i}'])
        escritor.writerow(cabecalho)

print("---------------------------------------------------------")
print("CÂMERA LIGADA: Faça o sinal e pressione a letra correspondente no teclado.")
print("Pressione 'q' para sair.")
print("---------------------------------------------------------")

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(frame_rgb)
    linha_dados = []

    if resultado.multi_hand_landmarks:
        for mao_pontos in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, mao_pontos, mp_maos.HAND_CONNECTIONS)
            for ponto in mao_pontos.landmark:
                linha_dados.extend([ponto.x, ponto.y])
                
    cv2.imshow("Coleta de Dados - Libras", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('q'):
        break
        
    # Aceita qualquer letra minúscula de 'a' até 'z'
    elif ord('a') <= tecla <= ord('z') and resultado.multi_hand_landmarks:
        letra = chr(tecla).upper() # Salva sempre em maiúsculo
        
        with open(arquivo_csv, mode='a', newline='') as f:
            escritor = csv.writer(f)
            linha_dados_completa = [letra] + linha_dados
            escritor.writerow(linha_dados_completa)
            
        print(f"[{letra}] capturada e salva no banco de dados!")

camera.release()
cv2.destroyAllWindows()