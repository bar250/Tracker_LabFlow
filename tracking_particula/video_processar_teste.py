import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def get_video_pairs(folder_path):
    """
    Obtém pares de vídeos na pasta, diferenciados por "VF" e "VL".
    
    Args:
        folder_path (str): Caminho da pasta onde os vídeos estão localizados.
    
    Returns:
        list: Lista de tuplas, onde cada tupla contém o caminho completo para um par de vídeos (VF, VL).
    """
    files = os.listdir(folder_path)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = [file for file in files if os.path.splitext(file)[1].lower() in video_extensions]

    # Criar um dicionário de pares com base no nome base dos vídeos
    video_pairs = {}
    for video in videos:
        base_name = re.sub(r'_VF|_VL', '', os.path.splitext(video)[0])
        if base_name not in video_pairs:
            video_pairs[base_name] = {}
        if '_VF' in video:
            video_pairs[base_name]['VF'] = os.path.join(folder_path, video)
        elif '_VL' in video:
            video_pairs[base_name]['VL'] = os.path.join(folder_path, video)

    # Filtrar apenas pares completos
    complete_pairs = [
        (paths['VF'], paths['VL']) for paths in video_pairs.values()
        if 'VF' in paths and 'VL' in paths
    ]

    if not complete_pairs:
        raise FileNotFoundError("Nenhum par de vídeos (VF e VL) encontrado na pasta.")

    return complete_pairs

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
    escala_largura_cm_per_pixel = largura_cm / roi_width_pixels
    escala_altura_cm_per_pixel = altura_cm / roi_height_pixels
    
    escala_largura_m_per_pixel = escala_largura_cm_per_pixel / 100
    escala_altura_m_per_pixel = escala_altura_cm_per_pixel / 100
    
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * start_time_seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, frame = cap.read()
    if not ret:
        print(f"Erro ao capturar o primeiro frame do vídeo {video_path}.")
        return

    height, width, _ = frame.shape

    # Redimensionar o frame para caber na tela
    screen_width, screen_height = 1280, 720  # Tamanho padrão da tela
    scaling_factor = min(screen_width / width, screen_height / height)
    frame_resized = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))

    # Selecionar ROI no frame redimensionado
    roi_resized = cv2.selectROI(f"Selecione a ROI para o vídeo {video_name}", frame_resized, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Selecione a ROI para o vídeo {video_name}")

    # Converter ROI redimensionada para as dimensões originais
    x_resized, y_resized, w_resized, h_resized = roi_resized
    x_roi = int(x_resized / scaling_factor)
    y_roi = int(y_resized / scaling_factor)
    w_roi = int(w_resized / scaling_factor)
    h_roi = int(h_resized / scaling_factor)

    if w_roi == 0 or h_roi == 0:
        print(f"ROI inválida para o vídeo {video_path}. O processamento será encerrado.")
        return

    escala_largura_m_per_pixel, escala_altura_m_per_pixel = calcular_escala(w_roi, h_roi)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w_roi, h_roi))

    trajectory = []

    with open(txt_file, 'w') as txt:
        txt.write("Tempo (s), Posição X (m), Posição Y (m)\n")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
            results = model(frame_roi)
            annotated_frame = results[0].plot()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x_pixels = (x1 + x2) / 2
                    center_y_pixels = (y1 + y2) / 2
                    
                    center_x_m = center_x_pixels * escala_largura_m_per_pixel
                    center_y_m = center_y_pixels * escala_altura_m_per_pixel
                    trajectory.append((center_x_m, center_y_m))

                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    current_time = current_frame / fps
                    txt.write(f"{current_time:.2f}, {center_x_m:.6f}, {center_y_m:.6f}\n")

            out.write(annotated_frame)

    cap.release()
    out.release()

    filtered_trajectory = remove_outliers_boxplot(trajectory)
    np.save(trajectory_file, filtered_trajectory)

    if filtered_trajectory:
        plot_trajectory(filtered_trajectory, w_roi, h_roi, trajectory_image, close_up_image)


def plot_trajectory(trajectory, desired_width, desired_height, trajectory_path="trajectory.png", close_up_path="trajectory_closeup.png"):
    """
    Plota a trajetória das partículas detectadas e salva dois gráficos:
    1. Um gráfico completo (dimensão do vídeo).
    2. Um gráfico com um close-up nas partículas detectadas.
    """
    if trajectory:
        x_vals, y_vals = zip(*trajectory)

        plt.figure(figsize=(10, 10))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
        plt.title("Trajetória da Partícula Detectada (Sem Outliers)")
        plt.xlabel("Posição X (m)")
        plt.ylabel("Posição Y (m)")
        plt.xlim(0, desired_width)
        plt.ylim(desired_height, 0)
        plt.savefig(trajectory_path)
        plt.show()

        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)

        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        min_x -= padding_x
        max_x += padding_x
        min_y -= padding_y
        max_y += padding_y

        plt.figure(figsize=(10, 10))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
        plt.title("Trajetória da Partícula Detectada (Close-up)")
        plt.xlabel("Posição X (m)")
        plt.ylabel("Posição Y (m)")
        plt.xlim(min_x, max_x)
        plt.ylim(max_y, min_y)
        plt.savefig(close_up_path)
        plt.show()

def process_video_pairs(video_pairs, model_path, output_base_folder, start_time_seconds=15):
    """
    Processa pares de vídeos e salva os resultados em pastas específicas para cada par.
    """
    for vf_path, vl_path in video_pairs:
        pair_name = re.sub(r'_VF|_VL', '', os.path.splitext(os.path.basename(vf_path))[0])
        pair_output_folder = os.path.join(output_base_folder, pair_name)

        if not os.path.exists(pair_output_folder):
            os.makedirs(pair_output_folder)

        print(f"Processando par: {vf_path} e {vl_path}")
        print(f"Pasta de saída: {pair_output_folder}")

        process_and_save_video(vf_path, model_path, pair_output_folder, start_time_seconds=start_time_seconds)
        process_and_save_video(vl_path, model_path, pair_output_folder, start_time_seconds=start_time_seconds)

folder_path = "./Vídeos"
model_path = "D:/Desktop/IA_barthos/runs/detect/train44/weights/best.pt"
output_base_folder = "./outputs"

video_pairs = get_video_pairs(folder_path)
process_video_pairs(video_pairs, model_path, output_base_folder, start_time_seconds=15)
