from ultralytics import YOLO
from pathlib import Path
# 1. Definir Caminhos
# O caminho para a imagem que será analisada
IMAGE_PATH = 'images_test', 'test.jpg' #

# O caminho para o seu modelo treinado (usaremos o PyTorch .pt, mas .onnx funcionaria)
MODELO_PATH = 'modelo/best.pt' 

# 2. Carregar o Modelo
try:
    model = YOLO(MODELO_PATH)
    print(f"✅ Modelo {MODELO_PATH} carregado com sucesso.")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    exit()

# 3. Executar a Inferência (Detecção)
print(f"🔍 Analisando a imagem: {IMAGE_PATH}...")

# O método predict() realiza a detecção
results = model.predict(
    source=IMAGE_PATH,
    save=True,      # Isso salva a imagem com as caixas delimitadoras e rótulos desenhados
    conf=0.5,       # Nível de confiança mínimo (50%)
    name='results_test1' # Nome da subpasta onde os resultados serão salvos dentro de 'runs/detect'
)

# 4. Exibir e Analisar os Resultados
print("\n--- Resultados Detalhados ---")

# 'results' é uma lista, pois o 'predict' pode aceitar múltiplas fontes (imagens/vídeos)
for r in results:
    boxes = r.boxes             # Acessa os bounding boxes (caixas delimitadoras)
    
    print(f"Total de detecções encontradas: {len(boxes)}")

    # Itera sobre cada detecção
    for i, box in enumerate(boxes):
        # Coordenadas da caixa (formato xyxy: canto superior esquerdo e canto inferior direito)
        coords = box.xyxy[0].tolist() 
        # ID da classe detectada
        class_id = int(box.cls[0].item())
        # Nome da classe
        class_name = model.names[class_id]
        # Score de confiança
        confidence = float(box.conf[0].item())

        print(f"  Detecção {i+1}:")
        print(f"    Classe: {class_name}")
        print(f"    Confiança: {confidence:.2f}") # Exibe com 2 casas decimais
        print(f"    Coordenadas: ({coords[0]:.0f}, {coords[1]:.0f}) a ({coords[2]:.0f}, {coords[3]:.0f})")

# Onde encontrar a imagem resultante (com as detecções desenhadas)
print("\n🖼️ Imagem com detecções salvas em: runs/detect/results_test1")