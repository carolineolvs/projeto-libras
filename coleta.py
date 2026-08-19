import cv2
import mediapipe as mp
import csv
import os

mp_maos = mp.solutions.hands
max_num_hands = 1
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.7, min_tracking_confidence=0.7)

camera = cv2.VideoCapture(0)

# O arquivo que servirá de banco de dados para a nossa IA
arquivo_csv = 'dados_libras.csv'

# Se o arquivo não existir, criamos ele com os cabeçalhos das colunas
if not os.path.exists(arquivo_csv):
    with open(arquivo_csv, mode='w', newline='') as f:
        escritor = csv.writer(f)
        # Primeira coluna é a Letra, as outras 42 são os pontos X e Y da mão
        cabecalho = ['letra']
        for i in range(21):
            cabecalho.extend([f'x{i}', f'y{i}'])
        escritor.writerow(cabecalho)

print("---------------------------------------------------------")
print("CÂMERA LIGADA: Faça o sinal e pressione 'a', 'b' ou 'c'.")
print("Pressione 'q' para sair.")
print("---------------------------------------------------------")

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        continue

    # O MediaPipe nos entrega as coordenadas x e y normalizadas (entre 0 e 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(frame_rgb)

    linha_dados = []

    if resultado.multi_hand_landmarks:
        for mao_pontos in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, mao_pontos, mp_maos.HAND_CONNECTIONS)
            
            # Pega todos os pontos da mão e coloca numa lista
            for ponto in mao_pontos.landmark:
                linha_dados.extend([ponto.x, ponto.y])
                
    cv2.imshow("Coleta de Dados - Libras", frame)
    tecla = cv2.waitKey(1) & 0xFF

    # Sair do programa
    if tecla == ord('q'):
        break
        
    # Se o usuário apertar A, B ou C, e houver uma mão na tela, salvamos no CSV
    elif tecla in [ord('a'), ord('b'), ord('c')] and resultado.multi_hand_landmarks:
        letra = chr(tecla).upper() # Transforma 'a' em 'A'
        
        with open(arquivo_csv, mode='a', newline='') as f:
            escritor = csv.writer(f)
            # Junta a Letra com as coordenadas capturadas e salva uma nova linha
            linha_dados_completa = [letra] + linha_dados
            escritor.writerow(linha_dados_completa)
            
        print(f"[{letra}] capturada e salva no banco de dados!")

camera.release()
cv2.destroyAllWindows()