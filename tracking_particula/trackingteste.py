import cv2
from ultralytics import YOLO

# Carrega o modelo
model = YOLO('D:/Desktop/IA_barthos/runs/detect/train44/weights/best.pt')

# Caminho do vídeo
video_path = 'D:/Desktop/tracking_final/tracking_particula/Vídeos/S1_T1_VF.MP4'
cap = cv2.VideoCapture(video_path)

# Cria uma janela e diz para o OpenCV que ela PODE ser redimensionada
cv2.namedWindow("Rastreamento", cv2.WINDOW_NORMAL)

# Define um tamanho inicial para caber na sua tela (ex: 1280x720)
cv2.resizeWindow("Rastreamento", 1280, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break # Fim do vídeo

    # Roda o rastreamento no frame atual
    # O persist=True é muito importante aqui para ele lembrar do ID da bolinha entre os frames!
    results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)

    # Pega o frame com as caixas e linhas desenhadas
    annotated_frame = results[0].plot()

    # Mostra na janela que configuramos
    cv2.imshow("Rastreamento", annotated_frame)

    # Aperte 'q' no teclado para fechar o vídeo no meio, se quiser
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
