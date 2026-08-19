import cv2
import mediapipe as mp

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

maos = mp_maos.Hands()
camera = cv2.VideoCapture(0)

print("Pressione a tecla 'q' na janela do vídeo para fechar.")

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(frame_rgb)

    if resultado.multi_hand_landmarks:
        for mao_pontos in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, mao_pontos, mp_maos.HAND_CONNECTIONS)
            
            # 1. Lista para guardar as coordenadas X e Y de todos os 21 pontos
            pontos = []
            for id, ponto in enumerate(mao_pontos.landmark):
                altura, largura, _ = frame.shape
                # Convertendo a proporção da IA para pixels reais da sua tela
                cx, cy = int(ponto.x * largura), int(ponto.y * altura)
                pontos.append((cx, cy))
            
            # 2. Lógica para identificar a Letra "A"
            if pontos:
                dedos_dobrados = 0
                
                # Verificando se a ponta do dedo está abaixo da articulação (Y maior = mais para baixo na tela)
                if pontos[8][1] > pontos[6][1]: dedos_dobrados += 1  # Dedo Indicador
                if pontos[12][1] > pontos[10][1]: dedos_dobrados += 1 # Dedo Médio
                if pontos[16][1] > pontos[14][1]: dedos_dobrados += 1 # Dedo Anelar
                if pontos[20][1] > pontos[18][1]: dedos_dobrados += 1 # Dedo Mindinho
                
                # Se os 4 dedos estiverem dobrados, assumimos que é a Letra A
                if dedos_dobrados == 4:
                    # Escreve "Letra A" na tela (em verde)
                    cv2.putText(frame, "Letra A", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Meu Tradutor de Libras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()