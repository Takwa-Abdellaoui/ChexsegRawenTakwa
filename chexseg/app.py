from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from analyse_cams import executer_analyse

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

# Vérification que le fichier existe bien dans chexseg_data/images
    allowed_dir = os.path.abspath("chexseg_data/images")
    expected_path = os.path.join(allowed_dir, file.filename)

    if not os.path.exists(expected_path):
     os.remove(save_path)
     return jsonify({"error": "Ce fichier ne semble pas être une radiographie thoracique autorisée."}), 400


    try:
        results, chart_base64 = executer_analyse(save_path)
        return jsonify({
            "filename": file.filename,
            "predictions": results,
            "chart": chart_base64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
