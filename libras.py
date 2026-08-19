import cv2
import mediapipe as mp

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

maos = mp_maos.Hands()
camera = cv2.VideoCapture(0)

print("Pressione a tecla 'q' na janela do vídeo para fechar o programa.")

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(frame_rgb)

    if resultado.multi_hand_landmarks:
        for mao_pontos in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(frame, mao_pontos, mp_maos.HAND_CONNECTIONS)

    cv2.imshow("Meu Tradutor de Libras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()