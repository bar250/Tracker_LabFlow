import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def ajustar_tempo_e_salvar_simples(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith("recognitions_") and ("_VF" in file or "_VL" in file):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, delimiter=",", encoding="latin1")
                
                df.columns = ["Tempo (s)", "Posição X (m)", "Posição Y (m)"]
                tempo_inicial = df.iloc[0, 0]
                df["Tempo (s)"] = (df["Tempo (s)"] - tempo_inicial).round(3)
                
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                output_file_path = os.path.join(output_folder, file)
                df.to_csv(output_file_path, index=False, sep=",", encoding="latin1")
                print(f"Arquivo processado: {output_file_path}")

def ajustar_tempo_e_salvar_completo(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(input_dir):
        vf_file = None
        vl_file = None
        
        for file in files:
            if file.startswith("recognitions_"):
                if "_VF" in file:
                    vf_file = os.path.join(root, file)
                elif "_VL" in file:
                    vl_file = os.path.join(root, file)
        
        if vf_file and vl_file:
            df_vf = pd.read_csv(vf_file, delimiter=",", encoding="latin1")
            df_vl = pd.read_csv(vl_file, delimiter=",", encoding="latin1")
            
            df_vf.columns = ["Tempo (s)", "X", "Y_VF"]
            df_vl.columns = ["Tempo (s)", "Z", "Y_VL"]
            
            tempo_comum = np.linspace(min(df_vf["Tempo (s)"].min(), df_vl["Tempo (s)"].min()),
                                      max(df_vf["Tempo (s)"].max(), df_vl["Tempo (s)"].max()),
                                      max(len(df_vf), len(df_vl)))
            
            df_vf_interp = pd.DataFrame({
                "Tempo (s)": tempo_comum,
                "X": np.interp(tempo_comum, df_vf["Tempo (s)"], df_vf["X"]),
                "Y_VF": np.interp(tempo_comum, df_vf["Tempo (s)"], df_vf["Y_VF"])
            })
            
            df_vl_interp = pd.DataFrame({
                "Tempo (s)": tempo_comum,
                "Z": np.interp(tempo_comum, df_vl["Tempo (s)"], df_vl["Z"]),
                "Y_VL": np.interp(tempo_comum, df_vl["Tempo (s)"], df_vl["Y_VL"])
            })
            
            df_final = df_vf_interp.merge(df_vl_interp, on="Tempo (s)")
            df_final["Y"] = (df_final["Y_VF"] + df_final["Y_VL"]) / 2
            df_final = df_final[["Tempo (s)", "X", "Y", "Z"]]
            
            relative_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            output_file_path = os.path.join(output_folder, "processed_recognitions.csv")
            df_final.to_csv(output_file_path, index=False, sep=",", encoding="latin1")
            print(f"Arquivo processado: {output_file_path}")


def plot_3d_graph(file_path, output_html):
    df = pd.read_csv(file_path, delimiter=",", encoding="latin1")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df["X"],
        y=df["Z"],
        z=df["Y"],
        mode='lines+markers',
        marker=dict(size=3, color=df["Y"], colorscale='Viridis'),
        line=dict(width=2)
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', range=[0, 0.20]),
            yaxis=dict(title='Z (m)', range=[0, 0.20]),
            zaxis=dict(title='Y (m)', range=[0.80, 0]),
            aspectratio=dict(x=1, y=1, z=4)
        ),
        title="Gráfico 3D Interativo da Trajetória"
    )
    
    fig.write_html(output_html)
    print(f"Gráfico salvo em: {output_html}")
    fig.show()

def calculate_velocity(file_path, step=1, remove_initial=True):
    df = pd.read_csv(file_path, delimiter=",", encoding="latin1")
    
    if "Tempo (s)" not in df.columns:
        raise ValueError("O arquivo CSV precisa conter uma coluna chamada 'Tempo (s)'.")
    
    df["Tempo (s)"] = df["Tempo (s)"].astype(float)
    df = df.reset_index(drop=True)
    
    velocities_x = np.full(len(df), np.nan)
    velocities_y = np.full(len(df), np.nan)
    velocities_z = np.full(len(df), np.nan)
    velocities_total = np.full(len(df), np.nan)
    
    for i in range(step, len(df)):
        dt = df.loc[i, "Tempo (s)"] - df.loc[i - step, "Tempo (s)"]
        if dt == 0:
            continue
        
        dx = df.loc[i, "X"] - df.loc[i - step, "X"]
        dy = df.loc[i, "Y"] - df.loc[i - step, "Y"]
        dz = df.loc[i, "Z"] - df.loc[i - step, "Z"]
        
        vx = dx / dt
        vy = dy / dt
        vz = dz / dt
        v_total = np.sqrt(vx**2 + vy**2 + vz**2)
        
        velocities_x[i] = vx
        velocities_y[i] = vy
        velocities_z[i] = vz
        velocities_total[i] = v_total
    
    df["Velocity_X"] = velocities_x
    df["Velocity_Y"] = velocities_y
    df["Velocity_Z"] = velocities_z
    df["Velocity_Total"] = velocities_total
    
    if remove_initial:
        df = df.iloc[step:].reset_index(drop=True)
    
    velocity_mean = np.nanmean(df["Velocity_Total"])
    print(f"Velocidade Média Geral: {velocity_mean:.4f} m/s")
    
    return df, velocity_mean

def remover_outliers_boxplot(df, colunas):
    """Remove outliers das colunas especificadas usando a metodologia do Boxplot (IQR)."""
    df_limpo = df.copy()
    
    for coluna in colunas:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        df_limpo = df_limpo[(df_limpo[coluna] >= limite_inferior) & (df_limpo[coluna] <= limite_superior)]
    
    return df_limpo

def plotar_grafico_velocidade(output_dir):
    velocities_file = None
    
    for file in os.listdir(output_dir):
        if file.startswith("velocities") and file.endswith(".csv"):
            velocities_file = os.path.join(output_dir, file)
            break
    
    if not velocities_file:
        print("Arquivo de velocidades não encontrado.")
        return
    
    df = pd.read_csv(velocities_file)

    # Removendo outliers das colunas de velocidade
    colunas_velocidade = ["Velocity_X", "Velocity_Y", "Velocity_Z", "Velocity_Total"]
    df = remover_outliers_boxplot(df, colunas_velocidade)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["Tempo (s)"], df["Velocity_X"], label="Velocity_X", linestyle="--", color="r")
    plt.plot(df["Tempo (s)"], df["Velocity_Y"], label="Velocity_Y", linestyle="--", color="g")
    plt.plot(df["Tempo (s)"], df["Velocity_Z"], label="Velocity_Z", linestyle="--", color="b")
    plt.plot(df["Tempo (s)"], df["Velocity_Total"], label="Velocity_Total", linewidth=2, color="k")
    
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade")
    plt.title("Componentes da Velocidade ao Longo do Tempo (Outliers Removidos)")
    plt.legend()
    plt.grid()
    
    output_file_path = os.path.join(output_dir, "velocity_plot.png")
    plt.savefig(output_file_path)
    print(f"Gráfico salvo em: {output_file_path}")
    
    plt.show()


def processar_pasta(pasta_input_path, pasta_output_path):
    """Processa uma única pasta executando todas as etapas necessárias."""
    print(f"Processando pasta: {pasta_input_path}")

    # Garantir que a pasta de saída existe
    os.makedirs(pasta_output_path, exist_ok=True)

    # Ajustar o tempo e salvar os arquivos processados
    ajustar_tempo_e_salvar_simples(pasta_input_path, pasta_output_path)
    ajustar_tempo_e_salvar_completo(pasta_output_path, pasta_output_path)

    # Caminho do arquivo CSV processado para análise e gráficos
    processed_file_path = os.path.join(pasta_output_path, "processed_recognitions.csv")

    if os.path.exists(processed_file_path):
        # Gerar gráfico 3D interativo
        output_html_path = os.path.join(pasta_output_path, "grafico_3D.html")
        plot_3d_graph(processed_file_path, output_html_path)

        # Calcular velocidades e exibir a velocidade média
        df_velocity, velocity_mean = calculate_velocity(processed_file_path, step=90, remove_initial=True)

        # Salvar o novo arquivo com velocidades calculadas
        velocity_file_path = os.path.join(pasta_output_path, "velocities.csv")
        df_velocity.to_csv(velocity_file_path, index=False, sep=",", encoding="latin1")

        print(f"Arquivo de velocidades salvo em: {velocity_file_path}")

        # Gerar gráfico de velocidade
        plotar_grafico_velocidade(pasta_output_path)
    else:
        print(f"Aviso: O arquivo {processed_file_path} não foi encontrado. Pulando para a próxima etapa.")

    print(f"Finalizado processamento da pasta: {pasta_input_path}\n")


def processar_todas_pastas(base_input_dir, base_output_dir):
    """Processa todas as pastas dentro do diretório base."""
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    for pasta in os.listdir(base_input_dir):
        pasta_input_path = os.path.join(base_input_dir, pasta)
        if os.path.isdir(pasta_input_path):
            pasta_output_path = os.path.join(base_output_dir, pasta)
            processar_pasta(pasta_input_path, pasta_output_path)

if __name__ == "__main__":
    base_input_directory = "D:/Dados_sedimentacao/Dados Barthos/C1- Feitos/Outputs"
    base_output_directory = "D:/Dados_sedimentacao/C1 - Outputs_Processados"
    
    processar_todas_pastas(base_input_directory, base_output_directory)
    print("Processamento concluído para todas as pastas.")
