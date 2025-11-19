from ultralytics import YOLO
from pathlib import Path
import sys # Usado para sair do script em caso de erro

# 1. Definir o Diretório Base
# Pega o diretório do arquivo atual (app.py) para construir caminhos relativos a ele.
BASE_DIR = Path(__file__).resolve().parent

# 2. Definir Caminhos dos Arquivos
# O 'BASE_DIR' garante que estamos na raiz do seu projeto.
IMAGE_PATH = BASE_DIR / 'images_test' / 'livro2.jpg' 
MODELO_PATH = BASE_DIR / 'modelo' / 'best.pt' 

# --- Verificações de Existência do Arquivo ---
# 3. Verificar o modelo
if not MODELO_PATH.exists():
    print(f"❌ ERRO CRÍTICO: Modelo não encontrado no caminho: {MODELO_PATH}")
    sys.exit(1) # Sai do programa

# 4. Verificar a Imagem de Teste (Causa do seu erro!)
if not IMAGE_PATH.exists():
    print("--- DETALHES DO ERRO DE CAMINHO ---")
    print(f"❌ ERRO CRÍTICO: Imagem de teste NÃO ENCONTRADA no caminho: {IMAGE_PATH}")
    print(f"Verifique se o arquivo 'test1.jpg' está em '{BASE_DIR / 'images_test'}'")
    sys.exit(1) # Sai do programa
# -----------------------------------------------

# 5. Carregar e Executar o Modelo
try:
    # Carrega o modelo PyTorch
    model = YOLO(str(MODELO_PATH)) 
    print(f"✅ Modelo {MODELO_PATH.name} carregado com sucesso.")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo YOLO: {e}")
    sys.exit(1)

print(f"🔍 Analisando a imagem: {IMAGE_PATH.name}...")

# Executar a Inferência (o método 'predict' agora usa o caminho absoluto garantido)
results = model.predict(
    source=str(IMAGE_PATH), # Converte o objeto Path de volta para string para o YOLO
    save=True,      
    conf=0.5,       
    name='results_test1' 
)

# ... (Restante do seu código para exibir os resultados)
print("\n--- Resultados Detalhados ---")
for r in results:
    boxes = r.boxes 
    print(f"Total de detecções encontradas: {len(boxes)}")
    for i, box in enumerate(boxes):
        coords = box.xyxy[0].tolist() 
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item())

        print(f"  Detecção {i+1}: Classe: {class_name}, Confiança: {confidence:.2f}")

print("\n🖼️ Imagem com detecções salvas em: runs/detect/")