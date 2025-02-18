import os
import imageio.v3 as iio
import numpy as np
import PIL.Image

def extract_frames_from_videos(base_path, output_path):
    """
    Percorre todas as pastas dentro de 'Microvídeos Cortados', extrai os frames dos vídeos e salva dentro de 'output'.
    """
    input_videos_path = os.path.join(base_path, "Microvídeos Cortados")
    os.makedirs(output_path, exist_ok=True)
    
    for folder in os.listdir(input_videos_path):
        folder_path = os.path.join(input_videos_path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        output_folder_path = os.path.join(output_path, folder)
        os.makedirs(output_folder_path, exist_ok=True)
        
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)
            
            if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue  # Ignora arquivos que não são vídeos
            
            print(f"📂 Processando: {video_path}")
            try:
                frames = iio.imread(video_path, plugin='pyav', index=None)  # Carregar todos os frames
            except Exception as e:
                print(f"⚠️ Erro ao carregar o vídeo: {e}")
                continue
            
            frame_count = 0
            
            for frame in frames:
                if frame is None or frame.size == 0:
                    print(f"⚠️ Frame {frame_count} de {video_file} está vazio ou corrompido e não foi salvo.")
                    continue
                
                image = PIL.Image.fromarray(frame)
                image = image.resize((320, 180), PIL.Image.LANCZOS)
                
                frame_filename = f"{os.path.splitext(video_file)[0]}_frame_{frame_count:04d}.jpg"
                frame_output_path = os.path.join(output_folder_path, frame_filename)
                
                try:
                    image.save(frame_output_path)  # Salvar frame como imagem redimensionada
                    print(f"✅ Frame {frame_filename} salvo com sucesso.")
                except Exception as e:
                    print(f"❌ Erro ao salvar {frame_filename}: {e}")
                
                frame_count += 1
            
    print("🚀 Processo concluído!")

if __name__ == "__main__":
    base_directory = os.getcwd()  # Diretório atual do script
    output_directory = os.path.join(base_directory, "output")
    os.makedirs(output_directory, exist_ok=True)
    extract_frames_from_videos(base_directory, output_directory)
