import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.utils.data import DataLoader, random_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import random
from collections import Counter
from collections import defaultdict

successful_frames = 0  # Contador de frames sem problemas
ignored_frames = 0  # Contador de frames ignorados

def calculate_class_accuracy(y_true, y_pred, num_classes):
    """
    Calcula a acurácia por classe.
    """
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1
    
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = class_correct[i] / class_total[i]
        else:
            class_accuracy[i] = 0.0  # Se não houver exemplos para a classe, define acurácia como 0.
    return class_accuracy

# Atualizar pesos dinamicamente
def adjust_loss_weights(class_accuracy, base_weight=1.0, max_weight=3.0, min_weight=0.5, smoothing=0.1):
    """
    Ajusta os pesos da perda com base na acurácia da classe.
    Aplica suavização para evitar variações bruscas e limita o peso máximo e mínimo.
    """
    global loss_weights  # Para manter pesos entre epochs
    
    if not hasattr(adjust_loss_weights, "loss_weights"):
        adjust_loss_weights.loss_weights = {cls: base_weight for cls in range(len(class_accuracy))}

    new_loss_weights = {}
    for cls, accuracy in class_accuracy.items():
        # Calcula o peso baseado na acurácia (quanto menor a acurácia, maior o peso)
        target_weight = base_weight / (accuracy + 1e-6)  # Evita divisão por zero

        # Limita os pesos dentro do intervalo definido
        target_weight = max(min_weight, min(target_weight, max_weight))

        # Suaviza a transição dos pesos para evitar mudanças bruscas
        new_loss_weights[cls] = (1 - smoothing) * adjust_loss_weights.loss_weights[cls] + smoothing * target_weight

    adjust_loss_weights.loss_weights = new_loss_weights  # Atualiza globalmente
    return new_loss_weights




# Criar os pesos para a perda
def create_class_weighted_loss(num_classes, class_weights):
    """
    Cria uma função de perda ponderada para CrossEntropyLoss.
    """
    weights = torch.tensor([class_weights.get(cls, 1.0) for cls in range(num_classes)], dtype=torch.float32).to(device)
    return nn.CrossEntropyLoss(weight=weights)

# Função para plotar a matriz de confusão
def plot_confusion_matrix(y_true, y_pred, class_names, epoch, phase="Validation"):
    """
    Plota e salva a matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{phase} Confusion Matrix (Epoch {epoch})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{phase.lower()}_confusion_matrix_epoch_{epoch}.png")
    plt.close()

# Caminho do dataset
dataset_path = "./Dataset_Balanceado"

# Criar o dataset com ImageFolder
asl_dataset = datasets.ImageFolder(dataset_path)

# Obter os nomes das classes dinamicamente
class_names = asl_dataset.classes

# Configuração do MediaPipe Hands
# Configuração do MediaPipe Holistic (detecção de pose, mãos e face)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.3
)

# Função para pré-processar imagens
def preprocess_image(image_np, attempt):
    """
    Realiza pré-processamento na imagem para melhorar a detecção.
    """
    h, w, _ = image_np.shape
    if attempt % 6 == 0:
        # Aumentar brilho e contraste
        image_preprocessed = cv2.convertScaleAbs(image_np, alpha=1.3, beta=30)
    elif attempt % 6 == 1:
        # Aplicar desfocagem Gaussian para reduzir ruídos
        image_preprocessed = cv2.GaussianBlur(image_np, (5, 5), 0)
    elif attempt % 6 == 2:
        # Ajustar brilho dinamicamente
        image_preprocessed = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX)
    elif attempt % 6 == 3:
        # Converter para escala de cinza e depois de volta para RGB
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        image_preprocessed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif attempt % 6 == 4:
        # Ajustar gama (correção de iluminação)
        gamma = 1.5  # Aumentar brilho
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image_preprocessed = cv2.LUT(image_np, look_up_table)
    elif attempt % 6 == 5:
        # Adicionar zoom out
        scale_factor = 0.8  # Zoom out de 20%
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized = cv2.resize(image_np, (new_w, new_h))
        # Adicionar bordas para manter o tamanho original
        top = (h - new_h) // 2
        bottom = h - new_h - top
        left = (w - new_w) // 2
        right = w - new_w - left
        image_preprocessed = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # Retornar a imagem original como fallback
        image_preprocessed = image_np
    return image_preprocessed



        # Variáveis de contagem global
total_images = 0
images_with_hands = 0
images_without_hands = 0
# Função para extrair landmarks e ROI

def extract_features_and_crop(image_np, debug=True, idx=None, max_attempts=24):
    """
    Processa a imagem usando MediaPipe Holistic para extrair:
      - Landmarks das mãos (esquerda e direita, se disponíveis)
      - Landmarks da pose referentes a ombros e braços
      - Um recorte (ROI) que engloba as mãos detectadas
      
    Retorna:
      - combined_hand_landmarks: vetor (concatenado) com os landmarks das mãos (ou None se nenhuma for detectada)
      - pose_features: vetor com os landmarks selecionados da pose (ombros e braços) ou None
      - cropped_region: imagem recortada com as mãos (se detectadas)
    """
    global total_images, images_with_hands, images_without_hands, successful_frames
    total_images += 1

    for attempt in range(max_attempts):
        image_preprocessed = preprocess_image(image_np, attempt)
        image_rgb = cv2.cvtColor(image_preprocessed, cv2.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)

        # Variáveis para armazenar os dados das mãos
        hand_boxes = []

        # Verifica se as duas mãos foram detectadas:
        if result.left_hand_landmarks and result.right_hand_landmarks:
            h, w, _ = image_np.shape

            # Processar mão esquerda:
            bbox_x_left = [int(lm.x * w) for lm in result.left_hand_landmarks.landmark]
            bbox_y_left = [int(lm.y * h) for lm in result.left_hand_landmarks.landmark]
            x_min_left, x_max_left = max(0, min(bbox_x_left)), min(w, max(bbox_x_left))
            y_min_left, y_max_left = max(0, min(bbox_y_left)), min(h, max(bbox_y_left))
            hand_boxes.append((x_min_left, y_min_left, x_max_left, y_max_left))
            left_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in result.left_hand_landmarks.landmark])

            # Processar mão direita:
            bbox_x_right = [int(lm.x * w) for lm in result.right_hand_landmarks.landmark]
            bbox_y_right = [int(lm.y * h) for lm in result.right_hand_landmarks.landmark]
            x_min_right, x_max_right = max(0, min(bbox_x_right)), min(w, max(bbox_x_right))
            y_min_right, y_max_right = max(0, min(bbox_y_right)), min(h, max(bbox_y_right))
            hand_boxes.append((x_min_right, y_min_right, x_max_right, y_max_right))
            right_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in result.right_hand_landmarks.landmark])

            # Combinar landmarks das duas mãos (cada mão deve ter 63 elementos → 21 pontos * 3)
            combined_hand_landmarks = np.concatenate([left_landmarks.flatten(), right_landmarks.flatten()])
        else:
            # Se alguma mão estiver ausente, ignora essa tentativa
            continue

        # Extração dos landmarks da pose (somente os índices selecionados)
        pose_features = None
        if result.pose_landmarks:
            selected_indices = [11, 12, 13, 14, 15, 16]
            pose_landmarks = []
            for i, lm in enumerate(result.pose_landmarks.landmark):
                if i in selected_indices:
                    pose_landmarks.append((lm.x, lm.y, lm.z))
            if len(pose_landmarks) == len(selected_indices):
                pose_features = np.array(pose_landmarks).flatten()

        # Se as mãos foram detectadas, gera a região recortada
        if hand_boxes:
            x_min = min([box[0] for box in hand_boxes])
            y_min = min([box[1] for box in hand_boxes])
            x_max = max([box[2] for box in hand_boxes])
            y_max = max([box[3] for box in hand_boxes])
            cropped_region = image_np[y_min:y_max, x_min:x_max]
            if cropped_region.size > 0 and cropped_region.shape[0] >= 20 and cropped_region.shape[1] >= 20:
                images_with_hands += 1
                successful_frames += 1  # Incrementa o contador de frames com sucesso
                #print(f"[SUCESSO] Frame {idx}: Processado corretamente! Total de frames bem-sucedidos: {successful_frames}")
                return (torch.tensor(combined_hand_landmarks, dtype=torch.float32),
                        torch.tensor(pose_features, dtype=torch.float32) if pose_features is not None else None,
                        cropped_region)
    images_without_hands += 1
    return None, None, None



# Dataset personalizado
# Carregador personalizado para carregar imagens como NumPy diretamente
def numpy_image_loader(path):
    """
    Carrega uma imagem do caminho fornecido e a converte diretamente em um array NumPy.
    """
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def compute_pairwise_distances(landmarks_np):
    """
    Recebe uma matriz NumPy de shape (n, 3) e retorna um vetor com as distâncias Euclidianas
    entre cada par de pontos (apenas a parte superior da matriz, sem a diagonal).
    """
    n = landmarks_np.shape[0]
    diff = landmarks_np[:, None, :] - landmarks_np[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_upper = np.triu_indices(n, k=1)
    distances = dists[i_upper]
    return distances

# Dataset personalizado
class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, crop_transform=None, mirror_probability=0.5):
        self.dataset = datasets.ImageFolder(dataset_path)
        self.crop_transform = crop_transform
        self.mirror_probability = mirror_probability

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if random.random() < self.mirror_probability:
            image_np = np.fliplr(image_np)

        hand_landmarks, pose_features, cropped_region = extract_features_and_crop(image_np, max_attempts=20, idx=idx)
        
        if hand_landmarks is None or pose_features is None or cropped_region is None:
            global ignored_frames
            ignored_frames += 1
            #print(f"[IGNORADO] Frame {idx}: Algum dos elementos não foi detectado. Total ignorados: {ignored_frames}")
            return None

        if self.crop_transform:
            cropped_region = self.crop_transform(Image.fromarray(cropped_region))

        # Combinar os landmarks: 126 (mãos) + 18 (pose) = 144 valores
        combined_landmarks = torch.cat((hand_landmarks, pose_features))  # Tensor shape (144,)

        # Converter para NumPy e remodelar para (48, 3) (48 pontos com 3 coordenadas cada)
        combined_landmarks_np = combined_landmarks.numpy().reshape(-1, 3)
        
        # Calcular as distâncias pairwise entre os 48 pontos (resultado: vetor com 1128 elementos)
        distances_np = compute_pairwise_distances(combined_landmarks_np)
        distances = torch.tensor(distances_np, dtype=torch.float32)

        # Retorna: landmarks combinados, distâncias calculadas, cropped_region e label
        return combined_landmarks, distances, cropped_region, label


# Transformações para as imagens recortadas
crop_transform = transforms.Compose([
    transforms.Resize((320, 180)),  # Redimensionar para o modelo
    transforms.ToTensor(),          # Converter para tensor
])


# Função de colação
def collate_fn(batch):
    batch = list(filter(None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    hand_landmarks, pose_features, cropped_regions, labels = zip(*batch)
    return (torch.stack(hand_landmarks) if hand_landmarks[0] is not None else torch.tensor([]),
            torch.stack(pose_features) if pose_features[0] is not None else torch.tensor([]),
            torch.stack(cropped_regions),
            torch.tensor(labels))


# Reduzir dataset com porcentagem
percentage = 100
asl_dataset = ASLDataset(dataset_path, crop_transform=crop_transform)
num_samples = int((percentage / 100) * len(asl_dataset))
indices = random.sample(range(len(asl_dataset)), num_samples)
subset_dataset = torch.utils.data.Subset(asl_dataset, indices)

train_size = int(0.9 * len(subset_dataset))
val_size = len(subset_dataset) - train_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# Modelo
class ASLClassifier(nn.Module):
    def __init__(self, landmark_size, distance_size, num_classes):
        super(ASLClassifier, self).__init__()
        self.cropped_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cropped_resnet.fc = nn.Identity()

        self.landmark_fc = nn.Sequential(
            nn.Linear(landmark_size, 256),  # landmark_size agora é 144
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.distance_fc = nn.Sequential(
            nn.Linear(distance_size, 256),  # distance_size agora é 1128
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # O total concatenado: 512 (cropped_resnet) + 128 (landmark_fc) + 128 (distance_fc) = 768
        self.fc1 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, landmarks, distances, cropped_hand):
        cropped_out = self.cropped_resnet(cropped_hand)
        landmarks_out = self.landmark_fc(landmarks)
        distances_out = self.distance_fc(distances)
        combined = torch.cat((cropped_out, landmarks_out, distances_out), dim=1)
        x = self.fc1(combined)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifier(landmark_size=144, distance_size=1128, num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Treinamento
epochs = 50

# Treinamento com ajuste dinâmico de pesos
best_val_accuracy = 0.0
class_weights = {cls: 1.0 for cls in range(len(class_names))}  # Inicializar pesos iguais para todas as classes.

for epoch in range(epochs):
    # Atualizar a função de perda com base nos pesos ajustados + label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([class_weights.get(cls, 1.0) for cls in range(len(class_names))], dtype=torch.float32).to(device),
        label_smoothing=0.1  # Suaviza a loss para evitar overfitting extremo em uma única classe
    )


    # Treinamento
    model.train()
    running_loss = 0.0
    for landmarks, distances, cropped_hands, labels in train_loader:
        if landmarks.nelement() == 0:
            continue
        landmarks, distances, cropped_hands, labels = (
            landmarks.to(device), distances.to(device), cropped_hands.to(device), labels.to(device)
        )
        optimizer.zero_grad()
        outputs = model(landmarks, distances, cropped_hands)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    # Validação
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    y_true_val, y_pred_val = [], []

    with torch.no_grad():
        for landmarks, distances, cropped_hands, labels in val_loader:
            if landmarks.nelement() == 0:
                continue
            landmarks, distances, cropped_hands, labels = (
                landmarks.to(device), distances.to(device), cropped_hands.to(device), labels.to(device)
            )
            outputs = model(landmarks, distances, cropped_hands)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Plotar matriz de confusão da validação
    plot_confusion_matrix(y_true_val, y_pred_val, class_names, epoch)

    # Calcular acurácia por classe e ajustar pesos
    class_accuracy = calculate_class_accuracy(y_true_val, y_pred_val, len(class_names))
    class_weights = adjust_loss_weights(class_accuracy)

    # Calcular e imprimir acurácia por classe
    class_correct = Counter()
    class_total = Counter()
    for true_label, pred_label in zip(y_true_val, y_pred_val):
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1

    print("\nClass-wise Accuracy:")
    for i, class_name in enumerate(class_names):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Class {class_name}: {accuracy:.2f}%")

    # Salvar o melhor modelo
    if epoch == 0 or val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_asl_model.pth")
        print(f"Best model saved at epoch {epoch + 1}")

print("\n===== Estatísticas de Detecção =====")
print(f"Total de imagens processadas: {total_images}")
print(f"Imagens com mãos detectadas: {images_with_hands} ({(images_with_hands / total_images) * 100:.2f}%)")
print(f"Imagens sem mãos detectadas: {images_without_hands} ({((images_without_hands) / total_images) * 100:.2f}%)")

print("Training complete.")