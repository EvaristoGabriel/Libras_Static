import cv2
import mediapipe as mp

# Inicializar o MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Criar o detector do Holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializar a webcam
cap = cv2.VideoCapture(0)

# Criar a janela antes do loop para evitar múltiplas janelas
cv2.namedWindow("MediaPipe Holistic - Pose e Mãos", cv2.WINDOW_NORMAL)

# Loop principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop se a captura falhar

    # Converter BGR para RGB para o MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem
    result = holistic.process(frame_rgb)

    # Converter RGB de volta para BGR para exibição no OpenCV
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Desenhar landmarks da pose, se detectados
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)
        )

    # Desenhar landmarks da mão esquerda, se detectados
    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3)
        )

    # Desenhar landmarks da mão direita, se detectados
    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
        )

    # Atualizar apenas a mesma janela
    cv2.imshow("MediaPipe Holistic - Pose e Mãos", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar tudo corretamente
cap.release()
cv2.destroyAllWindows()
