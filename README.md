Guia para o programa 

# Rastreamento de Partículas com YOLOv8

  Este projeto realiza o **processamento automático de pares de vídeos** (frontal e lateral) utilizando um modelo **YOLOv8** previamente treinado.  
  Ele detecta uma partícula (ou objeto de interesse), grava novos vídeos anotados, **gera a trajetória**, **remove outliers** e **salva os dados processados** em arquivos organizados.

---

## Estrutura esperada de diretórios

  Antes de rodar o script, garanta que sua estrutura de pastas siga o formato abaixo:
  '''
  projeto/
  main.py # Script principal
    Vídeos/ # Pasta contendo os vídeos a processar
      S1_T1_VF.mp4 # Vídeo da vista frontal
      S1_T1_VL.mp4 # Vídeo da vista lateral
      S2_T1_VF.mp4
      S2_T1_VL.mp4
      ...
    outputs/ # Resultados gerados automaticamente
      S1_T1/
        output_S1_T1_VF.mp4
        trajectory_S1_T1_VF.png
        trajectory_closeup_S1_T1_VF.png
        recognitions_S1_T1_VF.txt
        trajectory_S1_T1_VF.npy
        ...
  runs/
    detect/
      train44/
        weights/
          best.pt # Modelo YOLOv8 treinado
'''

##  Pré-requisitos

###  1. Instalação das dependências

  Certifique-se de ter o Python 3.9+ e instale as bibliotecas necessárias:
  
  pip install ultralytics opencv-python matplotlib numpy


###  2. Modelo YOLOv8 treinado

  Antes de usar o script, é necessário ter um modelo YOLOv8 previamente treinado para detectar a partícula ou objeto de interesse.
  
  Se ainda não treinou consulte aqui:
  **LINK DE GUIA DE TREINAMENTO DE REDES NEURAIS YOLOV8

##  Como Executar

  Basta rodar o script principal assim que as pastas de input estiverem organizadas como pedido:
  
  python main.py


##  O que o script faz?

  1.Busca automática de pares de vídeos
  
  2.Identifica vídeos com sufixos _VF (vista frontal) e _VL (vista lateral) dentro da pasta ./Vídeos.
  
  3.Seleção de ROI (Região de Interesse)
  
   Para cada vídeo, uma janela interativa do OpenCV permite selecionar a região onde ocorre o movimento da partícula.
    Essa ROI é usada para calcular a escala física (metros por pixel) e limitar o processamento.
  
  4.Processamento com YOLOv8
  
  O modelo é aplicado apenas na ROI.
  
  Para cada detecção, o centro do objeto é calculado e convertido em coordenadas reais (m).
  
  As informações são salvas em arquivos .txt e .npy.
  
  5.Remoção de outliers (Boxplot)
  
  Aplicação automática do método estatístico do boxplot para eliminar pontos fora do padrão da trajetória.
  
  6.Geração de resultados
  
  Vídeo anotado (output_*.mp4)
  
  Gráficos de trajetória (completo e close-up)
  
  Arquivos de coordenadas (.txt, .npy)

## Arquivos de saída
Dentro da pasta outputs/<NOME_DO_PAR>, são criados os seguintes arquivos:

Arquivo	Descrição
output_*.mp4	Vídeo anotado com as detecções YOLO
recognitions_*.txt	Lista de tempos e posições (em metros)
trajectory_*.npy	Trajetória filtrada (array NumPy)
trajectory_*.png	Trajetória completa
trajectory_closeup_*.png	Trajetória ampliada (close-up)

## Parâmetros principais

Você pode ajustar os caminhos e parâmetros no final do script:

folder_path = "./Vídeos"       # Pasta com os vídeos
model_path = "./caminho/para/seu/arquivo/best.pt"  # Modelo YOLO treinado
output_base_folder = "./outputs"  # Onde salvar os resultados
start_time_seconds = 15           # Tempo inicial para começar o processamento

## Estrutura do código
O código é modular e dividido em funções específicas:

Função	                  Responsabilidade
get_video_pairs()	        Identifica pares VF/VL na pasta
remove_outliers_boxplot()	Remove pontos fora do padrão (boxplot)
calcular_escala()	        Converte pixels → metros com base na ROI
process_and_save_video()	Processa e salva um único vídeo com YOLO
plot_trajectory()	        Gera gráficos da trajetória
process_video_pairs()	    Processa automaticamente todos os pares encontrados
