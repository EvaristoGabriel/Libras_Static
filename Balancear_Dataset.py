import os
import shutil
import random

# Definir caminhos
dataset_path = "Dataset"
balanced_path = "Dataset_Balanceado"

# Criar pasta balanceada se não existir
os.makedirs(balanced_path, exist_ok=True)

# Contar imagens em cada classe
class_counts = {}
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):  # Verifica se é uma pasta
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        class_counts[class_name] = len(image_files)

# Calcular média de imagens por classe
average_images = sum(class_counts.values()) // len(class_counts)
print(f"Média de imagens por classe: {average_images}")

# Criar a estrutura balanceada
for class_name, count in class_counts.items():
    original_class_path = os.path.join(dataset_path, class_name)
    balanced_class_path = os.path.join(balanced_path, class_name)

    os.makedirs(balanced_class_path, exist_ok=True)

    image_files = [f for f in os.listdir(original_class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Se a classe tem mais imagens que a média, apaga o excesso
    if count > average_images:
        selected_files = random.sample(image_files, average_images)  # Escolhe aleatoriamente quais manter
    else:
        # Se a classe tem menos imagens, duplica aleatoriamente até atingir a média
        selected_files = image_files.copy()
        while len(selected_files) < average_images:
            selected_files.append(random.choice(image_files))  # Duplica aleatoriamente

    # Copiar imagens para o novo dataset balanceado
    for i, file_name in enumerate(selected_files):
        src = os.path.join(original_class_path, file_name)
        ext = os.path.splitext(file_name)[1]  # Mantém a extensão do arquivo
        new_file_name = f"{class_name}_{i:04d}{ext}"  # Renomeia para evitar duplicatas
        dst = os.path.join(balanced_class_path, new_file_name)
        shutil.copy(src, dst)

print("✅ Dataset balanceado criado com sucesso!")
