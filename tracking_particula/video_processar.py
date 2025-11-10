import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os

def get_first_two_videos_in_folder(folder_path):
    """
    Obtém os dois primeiros arquivos de vídeo encontrados em uma pasta.
    
    Args:
        folder_path (str): Caminho da pasta onde o vídeo está localizado.
    
    Returns:
        tuple: Caminhos completos para os dois primeiros vídeos encontrados na pasta.
    """
    files = os.listdir(folder_path)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = [file for file in files if os.path.splitext(file)[1].lower() in video_extensions]
    
    if len(videos) < 2:
        raise FileNotFoundError("Menos de dois vídeos encontrados na pasta.")
    
    return os.path.join(folder_path, videos[0]), os.path.join(folder_path, videos[1])

def remove_outliers_boxplot(trajectory):
    """
    Remove outliers de uma trajetória usando a metodologia de boxplot.
    
    Args:
        trajectory (list): Lista de coordenadas (x, y).
    
    Returns:
        list: Trajetória sem os outliers.
    """
    x_vals, y_vals = zip(*trajectory)

    q1_x, q3_x = np.percentile(x_vals, [25, 75])
    iqr_x = q3_x - q1_x
    lower_bound_x = q1_x - 1.5 * iqr_x
    upper_bound_x = q3_x + 1.5 * iqr_x

    q1_y, q3_y = np.percentile(y_vals, [25, 75])
    iqr_y = q3_y - q1_y
    lower_bound_y = q1_y - 1.5 * iqr_y
    upper_bound_y = q3_y + 1.5 * iqr_y

    filtered_trajectory = [
        (x, y) for x, y in trajectory 
        if lower_bound_x <= x <= upper_bound_x and lower_bound_y <= y <= upper_bound_y
    ]

    return filtered_trajectory

# Função para calcular a escala de conversão de pixels para cm/m
def calcular_escala(roi_width_pixels, roi_height_pixels, largura_cm=20, altura_cm=80):
    """
    Calcula a escala de conversão de pixels para centímetros e metros com base nas dimensões reais da ROI.
    
    Args:
        roi_width_pixels (int): Largura da ROI em pixels.
        roi_height_pixels (int): Altura da ROI em pixels.
        largura_cm (float): Largura real da ROI em centímetros.
        altura_cm (float): Altura real da ROI em centímetros.
    
    Returns:
        tuple: Escalas de conversão para largura e altura (cm/pixel e m/pixel).
    """
    escala_largura_cm_per_pixel = largura_cm / roi_width_pixels  # cm por pixel para largura
    escala_altura_cm_per_pixel = altura_cm / roi_height_pixels   # cm por pixel para altura
    
    # Converter para metros por pixel
    escala_largura_m_per_pixel = escala_largura_cm_per_pixel / 100  # metros por pixel (largura)
    escala_altura_m_per_pixel = escala_altura_cm_per_pixel / 100    # metros por pixel (altura)
    
    return escala_largura_m_per_pixel, escala_altura_m_per_pixel

def process_and_save_video(video_path, model_path, output_folder, start_time_seconds=15):
    """
    Processa um vídeo com YOLOv8, salva os frames anotados em um novo arquivo de vídeo,
    armazena as posições detectadas, remove os outliers por boxplot, plota e salva a trajetória no final.
    Também armazena o tempo e as coordenadas de cada partícula reconhecida em um arquivo de texto,
    com base no tempo relativo ao vídeo.
    
    Args:
        video_path (str): Caminho do vídeo de entrada.
        model_path (str): Caminho para o modelo YOLOv8.
        output_folder (str): Pasta onde salvar os arquivos processados.
        start_time_seconds (int): Tempo de início em segundos para começar o processamento.
    """
    # Certifique-se de que o diretório de saída existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    output_video = os.path.join(output_folder, f"output_{video_name}.mp4")
    trajectory_file = os.path.join(output_folder, f"trajectory_{video_name}.npy")
    trajectory_image = os.path.join(output_folder, f"trajectory_{video_name}.png")
    close_up_image = os.path.join(output_folder, f"trajectory_closeup_{video_name}.png")
    txt_file = os.path.join(output_folder, f"recognitions_{video_name}.txt")
    
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo {video_path}.")
        return

    # Obter o FPS e calcular o frame de início com base no tempo de início
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * start_time_seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Pular para o frame correspondente ao tempo de início

    # Seleção manual de ROI para cada vídeo
    ret, frame = cap.read()
    if not ret:
        print(f"Erro ao capturar o primeiro frame do vídeo {video_path}.")
        return

    # Redimensionar o frame para caber na tela
    screen_width, screen_height = 1280, 720  # Tamanho padrão da tela, ajuste conforme necessário
    height, width, _ = frame.shape
    scaling_factor = min(screen_width / width, screen_height / height)
    frame_resized = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))

    # Selecionar ROI manualmente no frame redimensionado
    roi_resized = cv2.selectROI(f"Selecione a ROI para o vídeo {video_name}", frame_resized, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Selecione a ROI para o vídeo {video_name}")

    # Converter ROI redimensionada de volta para as dimensões originais
    x_resized, y_resized, w_resized, h_resized = roi_resized
    x_roi = int(x_resized / scaling_factor)
    y_roi = int(y_resized / scaling_factor)
    w_roi = int(w_resized / scaling_factor)
    h_roi = int(h_resized / scaling_factor)

    if w_roi == 0 or h_roi == 0:
        print(f"ROI inválida para o vídeo {video_path}. O processamento será encerrado.")
        return

    # Calcular a escala da ROI com base nos 20 cm x 115 cm
    escala_largura_m_per_pixel, escala_altura_m_per_pixel = calcular_escala(w_roi, h_roi)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Reiniciar a leitura a partir do tempo definido

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w_roi, h_roi))

    trajectory = []

    with open(txt_file, 'w') as txt:
        txt.write("Tempo (s), Posição X (m), Posição Y (m)\n")  # Cabeçalho do arquivo .txt

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print(f"Fim do vídeo ou erro ao ler frame no vídeo {video_path}.")
                break

            # Aplicar a ROI ao frame
            frame_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

            # Passar o frame ROI ao modelo
            results = model(frame_roi)

            # Anotar o frame
            annotated_frame = results[0].plot()

            # Coletar as coordenadas centrais das boxes detectadas
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x_pixels = (x1 + x2) / 2
                    center_y_pixels = (y1 + y2) / 2
                    
                    # Converter coordenadas de pixels para metros
                    center_x_m = center_x_pixels * escala_largura_m_per_pixel
                    center_y_m = center_y_pixels * escala_altura_m_per_pixel
                    trajectory.append((center_x_m, center_y_m))

                    # Tempo em relação ao vídeo
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Frame atual no vídeo
                    current_time = current_frame / fps  # Tempo decorrido no vídeo em segundos
                    txt.write(f"{current_time:.2f}, {center_x_m:.6f}, {center_y_m:.6f}\n")

            out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Processamento concluído! Vídeo salvo em: {output_video}")
    print(f"Dados de reconhecimento salvos em: {txt_file}")

    filtered_trajectory = remove_outliers_boxplot(trajectory)

    # Salvar o vetor de trajetória filtrada
    np.save(trajectory_file, filtered_trajectory)
    print(f"Vetor de trajetória salvo em: {trajectory_file}")

    if filtered_trajectory:
        plot_trajectory(filtered_trajectory, w_roi, h_roi, trajectory_image, close_up_image)

def plot_trajectory(trajectory, desired_width, desired_height, trajectory_path="trajectory.png", close_up_path="trajectory_closeup.png"):
    """
    Plota a trajetória das partículas detectadas e salva dois gráficos:
    1. Um gráfico completo (dimensão do vídeo).
    2. Um gráfico com um close-up nas partículas detectadas.
    
    Args:
        trajectory (list): Lista de coordenadas (x, y).
        desired_width (int): Largura do vídeo para ajuste dos eixos.
        desired_height (int): Altura do vídeo para ajuste dos eixos.
        trajectory_path (str): Caminho para salvar o gráfico completo.
        close_up_path (str): Caminho para salvar o gráfico focado nas partículas.
    """
    if trajectory:
        x_vals, y_vals = zip(*trajectory)

        # --- Gráfico completo (dimensão total do vídeo) ---
        plt.figure(figsize=(10, 10))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
        plt.title("Trajetória da Partícula Detectada (Sem Outliers)")
        plt.xlabel("Posição X (m)")
        plt.ylabel("Posição Y (m)")
        plt.xlim(0, desired_width)
        plt.ylim(desired_height, 0)  # Inverter o eixo Y
        plt.savefig(trajectory_path)
        print(f"Gráfico completo salvo em: {trajectory_path}")
        plt.show()

        # --- Gráfico focado nas partículas detectadas ---
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)

        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        min_x -= padding_x
        max_x += padding_x
        min_y -= padding_y
        max_y += padding_y

        min_x = max(min_x, 0)
        max_x = min(max_x, desired_width)
        min_y = max(min_y, 0)
        max_y = min(max_y, desired_height)

        plt.figure(figsize=(10, 10))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
        plt.title("Trajetória da Partícula Detectada (Close-up)")
        plt.xlabel("Posição X (m)")
        plt.ylabel("Posição Y (m)")
        plt.xlim(min_x, max_x)
        plt.ylim(max_y, min_y)  # Inverter o eixo Y para o sistema de coordenadas da imagem
        plt.savefig(close_up_path)
        print(f"Gráfico close-up salvo em: {close_up_path}")
        plt.show()

def load_and_plot_trajectory(trajectory_file, desired_width, desired_height, trajectory_path="trajectory.png", close_up_path="trajectory_closeup.png"):
    """
    Carrega o vetor de trajetória salvo e plota o gráfico.
    
    Args:
        trajectory_file (str): Caminho do arquivo .npy contendo a trajetória salva.
        desired_width (int): Largura do vídeo para ajuste dos eixos.
        desired_height (int): Altura do vídeo para ajuste dos eixos.
        trajectory_path (str): Caminho para salvar o gráfico completo.
        close_up_path (str): Caminho para salvar o gráfico focado nas partículas.
    """
    trajectory = np.load(trajectory_file, allow_pickle=True)

    plot_trajectory(trajectory, desired_width, desired_height, trajectory_path, close_up_path)

# --- Processar dois vídeos e salvar os resultados ---
folder_path = "./Vídeos"
model_path = "D:/Desktop/IA_barthos/runs/detect/train44/weights/best.pt"
output_folder = "./outputs"

# Obter os dois primeiros vídeos da pasta
video1, video2 = get_first_two_videos_in_folder(folder_path)

# Processar os dois vídeos separadamente, com seleção de ROI para cada um
process_and_save_video(video1, model_path, output_folder, start_time_seconds=15)
process_and_save_video(video2, model_path, output_folder, start_time_seconds=15)
