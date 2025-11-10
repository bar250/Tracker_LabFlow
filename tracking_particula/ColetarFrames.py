import cv2
import os
import screeninfo

def get_first_video_path(videos_folder):
    """
    Encontra o primeiro arquivo de vídeo na pasta especificada.
    """
    files = os.listdir(videos_folder)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in files if f.lower().endswith(video_extensions)]
    
    if len(video_files) == 0:
        print("Nenhum vídeo encontrado na pasta.")
        return None
    
    return os.path.join(videos_folder, video_files[0])

def get_screen_resolution():
    """
    Obtém a resolução da tela do computador.
    """
    screen = screeninfo.get_monitors()[0]
    return screen.width, screen.height

def resize_frame_to_screen(frame, screen_width, screen_height):
    """
    Redimensiona o frame para caber na tela mantendo a proporção.
    """
    frame_height, frame_width = frame.shape[:2]
    scale_width = screen_width / frame_width
    scale_height = screen_height / frame_height
    scale = min(scale_width, scale_height)
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    return cv2.resize(frame, (new_width, new_height))

def select_roi(video_path):
    """
    Seleciona a Região de Interesse (ROI) no primeiro frame do vídeo, com redimensionamento para caber na tela.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return None

    screen_width, screen_height = get_screen_resolution()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro ao ler o frame.")
            return None

        resized_frame = resize_frame_to_screen(frame, screen_width, screen_height)
        
        cv2.imshow("Navegue pelos frames (tecle 'n' para próximo frame ou 's' para selecionar a ROI)", resized_frame)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('n'):
            continue
        elif key == ord('s'):
            roi = cv2.selectROI("Selecione a área de interesse (ROI)", resized_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            cap.release()
            
            scale_x = frame.shape[1] / resized_frame.shape[1]
            scale_y = frame.shape[0] / resized_frame.shape[0]
            
            x, y, w, h = roi
            return int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

def select_frame_range(video_path):
    """
    Exibe um slider para selecionar o início e o fim do vídeo para a coleta de frames.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def update_frame_position(x):
        cap.set(cv2.CAP_PROP_POS_FRAMES, x)
        ret, frame = cap.read()
        if ret:
            resized_frame = resize_frame_to_screen(frame, screen_width, screen_height)
            cv2.imshow('Selecione o intervalo de frames', resized_frame)

    screen_width, screen_height = get_screen_resolution()

    cv2.namedWindow('Selecione o intervalo de frames')
    cv2.createTrackbar('Início', 'Selecione o intervalo de frames', 0, total_frames - 1, update_frame_position)
    cv2.createTrackbar('Fim', 'Selecione o intervalo de frames', total_frames - 1, total_frames - 1, update_frame_position)

    update_frame_position(0)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('s'):
            start_frame = cv2.getTrackbarPos('Início', 'Selecione o intervalo de frames')
            end_frame = cv2.getTrackbarPos('Fim', 'Selecione o intervalo de frames')
            cv2.destroyAllWindows()
            cap.release()
            return start_frame, end_frame
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            return None, None

def extract_frames(video_path, output_folder, start_frame, end_frame, roi=None, frame_interval=1):
    """
    Extrai frames de um vídeo dentro do intervalo selecionado e salva-os.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    saved_frame_count = 0

    while frame_count <= end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{saved_frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extração concluída! {saved_frame_count} frames foram salvos na pasta '{output_folder}'.")

if __name__ == "__main__":
    videos_folder = 'Vídeos'
    output_folder = 'Frames'

    video_path = get_first_video_path(videos_folder)

    if video_path:
        start_frame, end_frame = select_frame_range(video_path)
        if start_frame is not None and end_frame is not None:
            print(f"Selecionado intervalo de frames: Início = {start_frame}, Fim = {end_frame}")

            roi = select_roi(video_path)

            extract_frames(video_path, output_folder, start_frame, end_frame, roi=roi, frame_interval=10)
