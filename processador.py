import cv2
import pytesseract
import json
import numpy as np


# --- CONFIGURAÇÃO IMPORTANTE (SÓ PARA WINDOWS) ---
# Se você está no Windows, descomente (apague o #) a linha abaixo
# e coloque o caminho exato onde o Tesseract foi instalado.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- O MAPA DO TESOURO (AS COORDENADAS) ---
# Adicionamos os 4 círculos de porcentagem.
# O nome deles começa com "circulo_" para podermos identificá-los no código.
MAPA_DE_COORDENADAS = {
    # --- Círculos (Esquerda) ---
    "circulo_time1_dribles":      (300, 300, 100, 60), # X Y L A
    "circulo_time1_precisao_fin": (300, 560, 100, 60), 
    "circulo_time1_precisao_pas": (300, 820, 100, 60),

    # --- Círculos (Direita) ---
    "circulo_time2_dribles":      (1520, 300, 100, 60),
    "circulo_time2_precisao_fin": (1520, 560, 100, 60),
    "circulo_time2_precisao_pas": (1520, 820, 100, 60),
    
    # --- Tabela Central (Time 1 - Esquerda) ---
    "tabela_time1_posse_bola":         (660, 240, 47, 53), #ERA 60
    "tabela_time1_recuperacao_bola":   (660, 300, 40, 40),
    "tabela_time1_finalizacoes":       (660, 343, 40, 48),
    "tabela_time1_gols_esperados":     (660, 391, 60, 49),
    "tabela_time1_passes":             (660, 441, 60, 47),
    "tabela_time1_divididas":          (660, 490, 60, 48),
    "tabela_time1_divididas_ganhas":   (660, 539, 60, 48),
    "tabela_time1_interceptacoes":     (660, 588, 60, 48),
    "tabela_time1_defesas":            (660, 636, 60, 48),
    "tabela_time1_faltas_cometidas":   (660, 686, 40, 48),
    "tabela_time1_impedimentos":       (660, 735, 60, 48),
    "tabela_time1_escanteios":         (660, 784, 40, 48),
    "tabela_time1_faltas":             (660, 833, 40, 48),
    "tabela_time1_penaltis":           (660, 882, 40, 48),
    "tabela_time1_cartoes_amarelos":   (648, 940, 52, 40),

    # --- Tabela Central (Time 2 - Direita) ---
    "tabela_time2_posse_bola":         (1220, 240, 49, 51),
    "tabela_time2_recuperacao_bola":   (1220, 300, 40, 40),
    "tabela_time2_finalizacoes":       (1220, 343, 40, 48),
    "tabela_time2_gols_esperados":     (1180, 392, 80, 48),
    "tabela_time2_passes":             (1220, 441, 60, 47),
    "tabela_time2_divididas":          (1220, 490, 60, 48),
    "tabela_time2_divididas_ganhas":   (1200, 539, 60, 48),
    "tabela_time2_interceptacoes":     (1220, 588, 60, 48),
    "tabela_time2_defesas":            (1220, 636, 60, 48),
    "tabela_time2_faltas_cometidas":   (1220, 686, 40, 48),
    "tabela_time2_impedimentos":       (1200, 735, 72, 48),
    "tabela_time2_escanteios":         (1220, 784, 40, 48),
    "tabela_time2_faltas":             (1200, 833, 68, 47),
    "tabela_time2_penaltis":           (1220, 882, 40, 48),
    "tabela_time2_cartoes_amarelos":   (1211, 940, 58, 40),
}

# --- NOVA FUNÇÃO DE PRÉ-PROCESSAMENTO (A "LINHA DE MONTAGEM") ---
def processar_imagem(caminho_da_imagem):
    dados_finais = {}

    # 1) Carrega
    original = cv2.imread(caminho_da_imagem)
    if original is None:
        print(f"Erro: não foi possível carregar a imagem '{caminho_da_imagem}'.")
        return {}

    # 2) Normaliza para 1920x1080 (sem distorção se a imagem já for 16:9, como 1600x900)
    TARGET_W, TARGET_H = 1920, 1080
    if (original.shape[1], original.shape[0]) != (TARGET_W, TARGET_H):
        original = cv2.resize(original, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    # 3) Tons de cinza
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 4) Configs do Tesseract
    config_circulos     = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789%'
    config_multi_digito = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789,.'
    config_um_digito    = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'

    zeros_time1_sensiveis = [
        "tabela_time1_faltas_cometidas", "tabela_time1_impedimentos",
        "tabela_time1_faltas", "tabela_time1_penaltis", "tabela_time1_cartoes_amarelos"
    ]
    outros_um_digito = [
        "tabela_time1_recuperacao_bola", "tabela_time1_finalizacoes", "tabela_time1_divididas_ganhas",
        "tabela_time1_interceptacoes", "tabela_time1_defesas", "tabela_time1_escanteios",
        "tabela_time2_recuperacao_bola", "tabela_time2_finalizacoes", "tabela_time2_divididas_ganhas",
        "tabela_time2_interceptacoes", "tabela_time2_defesas", "tabela_time2_faltas_cometidas",
        "tabela_time2_impedimentos", "tabela_time2_escanteios", "tabela_time2_faltas",
        "tabela_time2_penaltis", "tabela_time2_cartoes_amarelos"
    ]

    H, W = gray.shape[:2]

    for nome, (x, y, w, h) in MAPA_DE_COORDENADAS.items():
        # Clipping defensivo (evita recortes fora da imagem)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)

        if x2 <= x1 or y2 <= y1:
            dados_finais[nome] = ""
            continue

        rec = gray[y1:y2, x1:x2]
        if rec.size == 0:
            dados_finais[nome] = ""
            continue

        # Upscaling do recorte (zoom para OCR)
        rec_up = cv2.resize(rec, (rec.shape[1]*2, rec.shape[0]*2), interpolation=cv2.INTER_CUBIC)

        # Pré-processamento conforme o tipo
        if nome.startswith("circulo_"):
            _, rec_proc = cv2.threshold(rec_up, 170, 255, cv2.THRESH_BINARY)
            config = config_circulos
        elif nome in zeros_time1_sensiveis:
            _, rec_proc = cv2.threshold(rec_up, 140, 255, cv2.THRESH_BINARY_INV)
            config = config_um_digito
        elif nome in outros_um_digito:
            _, rec_proc = cv2.threshold(rec_up, 160, 255, cv2.THRESH_BINARY_INV)
            config = config_um_digito
        else:
            _, rec_proc = cv2.threshold(rec_up, 170, 255, cv2.THRESH_BINARY_INV)
            config = config_multi_digito

        texto = pytesseract.image_to_string(rec_proc, config=config)
        valor = texto.strip()
        dados_finais[nome] = valor

    return dados_finais

if __name__ == "__main__":
    arquivo_imagem = 'jogo2.png'  # troque pelo seu arquivo para testar
    estatisticas = processar_imagem(arquivo_imagem)
    print("\n--- RESULTADO FINAL (FORMATO JSON) ---")
    print(json.dumps(estatisticas, indent=4, ensure_ascii=False))

