import cv2
import pytesseract
import json
import numpy as np


# --- CONFIGURAÇÃO IMPORTANTE (SÓ PARA WINDOWS) ---
# Se você está no Windows, descomente (apague o #) a linha abaixo
# e coloque o caminho exato onde o Tesseract foi instalado.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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
    imagem_original = cv2.imread(caminho_da_imagem)
    if imagem_original is None:
        print(f"Erro: não foi possível carregar a imagem '{caminho_da_imagem}'.")
        return {}
        
    imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

    # Configurações Otimizadas
    config_circulos = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789%'
    config_multi_digito = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789,.'
    config_um_digito = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
    config_palavra_robusta = r'--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789'

    # Listas de Controle Finais
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

    for nome_da_estatistica, (x, y, w, h) in MAPA_DE_COORDENADAS.items():
        recorte_cinza = imagem_cinza[y:y+h, x:x+w]

        # A "Lupa Digital" (Upscaling) que você sugeriu
        recorte_ampliado = cv2.resize(recorte_cinza, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # Lógica de Calibragem Híbrida
        if nome_da_estatistica.startswith("circulo_"):
            valor_corte = 170
            _, recorte_processado = cv2.threshold(recorte_ampliado, valor_corte, 255, cv2.THRESH_BINARY)
            config_atual = config_circulos
        elif nome_da_estatistica in zeros_time1_sensiveis:
            valor_corte = 140
            _, recorte_processado = cv2.threshold(recorte_ampliado, valor_corte, 255, cv2.THRESH_BINARY_INV)
            config_atual = config_um_digito
        elif nome_da_estatistica in outros_um_digito:
            valor_corte = 160
            _, recorte_processado = cv2.threshold(recorte_ampliado, valor_corte, 255, cv2.THRESH_BINARY_INV)
            config_atual = config_um_digito
        else:
            valor_corte = 170
            _, recorte_processado = cv2.threshold(recorte_ampliado, valor_corte, 255, cv2.THRESH_BINARY_INV)
            config_atual = config_multi_digito
        
        # Ative para a calibragem final entre diferentes imagens
        # cv2.imshow(f"Debug: {nome_da_estatistica}", recorte_processado)
        # cv2.waitKey(0)

        texto_extraido = pytesseract.image_to_string(recorte_processado, config=config_atual)
        valor_limpo = texto_extraido.strip()
        dados_finais[nome_da_estatistica] = valor_limpo
        
    cv2.destroyAllWindows()
    return dados_finais

# --- O "BOTÃO DE PLAY" ---
if __name__ == "__main__":
    arquivo_imagem = 'jogo2.png'
    estatisticas = processar_imagem(arquivo_imagem)
    print("\n--- RESULTADO FINAL (FORMATO JSON) ---")
    print(json.dumps(estatisticas, indent=4))