import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

# 1. Carregar os dados e treinar o modelo
dados = pd.read_csv('dados_libras.csv')
X = dados.drop('letra', axis=1)
y = dados['letra']

modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(X, y)

# 2. Configurar MediaPipe e Câmera
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

camera = cv2.VideoCapture(0)

def desenhar_hud(frame, letra="-", certeza=0, mao_detectada=False):
    # Criar uma cópia para aplicar transparência no topo
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (20, 20, 20), -1)
    
    # Aplica transparência (0.6 imagem original + 0.4 painel escuro)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # 1. Indicador de Status (LED)
    cor_led = (0, 255, 120) if mao_detectada else (100, 100, 100)
    cv2.circle(frame, (35, 40), 10, cor_led, -1)
    
    texto_status = "MAO DETECTADA" if mao_detectada else "PROCURANDO MAO..."
    cv2.putText(frame, texto_status, (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    if mao_detectada:
        # 2. Barra de Confiança (%)
        cv2.putText(frame, "CONFIANCA:", (55, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        largura_barra = int((certeza / 100) * 150)
        cv2.rectangle(frame, (140, 65), (290, 78), (60, 60, 60), -1)  # Fundo da barra
        cv2.rectangle(frame, (140, 65), (140 + largura_barra, 78), (0, 255, 120), -1)  # Preenchimento
        cv2.putText(frame, f"{certeza:.0f}%", (300, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 3. Cartão de Destaque da Letra
        cv2.rectangle(frame, (frame.shape[1] - 190, 15), (frame.shape[1] - 15, 85), (0, 180, 255), -1)
        cv2.rectangle(frame, (frame.shape[1] - 190, 15), (frame.shape[1] - 15, 85), (255, 255, 255), 2)
        cv2.putText(frame, "LETRA", (frame.shape[1] - 175, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, letra, (frame.shape[1] - 125, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        continue

    # Espelha o vídeo para a movimentação ficar intuitiva
    frame = cv2.flip(frame, 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(frame_rgb)

    mao_detectada = False
    letra = "-"
    certeza = 0

    if resultado.multi_hand_landmarks:
        mao_detectada = True
        for mao_pontos in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, mao_pontos, mp_maos.HAND_CONNECTIONS)
            
            linha_dados = []
            for ponto in mao_pontos.landmark:
                linha_dados.extend([ponto.x, ponto.y])
            
            letra = modelo.predict([linha_dados])[0]
            certeza = np.max(modelo.predict_proba([linha_dados])) * 100

    # Desenha o painel estilizado
    desenhar_hud(frame, letra, certeza, mao_detectada)

    cv2.imshow("Tradutor de Libras - HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()