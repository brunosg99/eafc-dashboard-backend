from flask import Flask, request, jsonify
from flask_cors import CORS
import processador # Importamos nosso script original
import os
import uuid # Para gerar nomes de arquivo únicos

app = Flask(__name__)
CORS(app, resources={r"/processar-imagem": {"origins": "https://eafc-dashboard-mvp.vercel.app"}}) # Habilita a comunicação entre domínios diferentes

# Define uma pasta temporária para salvar os uploads
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/processar-imagem', methods=['POST'])
def processar_endpoint():
    if 'imagem' not in request.files:
        return jsonify({"erro": "Nenhum arquivo de imagem enviado"}), 400
    
    file = request.files['imagem']
    
    if file.filename == '':
        return jsonify({"erro": "Nenhum arquivo selecionado"}), 400

    if file:
        # Gera um nome de arquivo seguro e único
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Chama a nossa função de processamento que já está perfeita!
            estatisticas = processador.processar_imagem(filepath)
            
            # Remove o arquivo temporário depois de processar
            os.remove(filepath)
            
            # Retorna o JSON com as estatísticas
            return jsonify(estatisticas)
            
        except Exception as e:
            # Em caso de erro, remove o arquivo e informa o erro
            os.remove(filepath)
            return jsonify({"erro": f"Ocorreu um erro no processamento: {str(e)}"}), 500

if __name__ == '__main__':

    app.run(debug=True)
