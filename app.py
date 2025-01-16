from flask import Flask, request, jsonify, send_file
import os
import subprocess
import tempfile

app = Flask(__name__)

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    """
    Recibe un archivo PDF, lo procesa con OCR y devuelve el texto como respuesta.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    pdf_file = request.files['file']

    # Verificar si es un archivo válido
    if pdf_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Crear carpetas temporales para procesar el archivo
    with tempfile.TemporaryDirectory() as temp_dir:
        input_folder = os.path.join(temp_dir, "input")
        output_folder = os.path.join(temp_dir, "output")
        os.makedirs(input_folder)
        os.makedirs(output_folder)

        # Guardar el archivo PDF en la carpeta de entrada
        input_path = os.path.join(input_folder, pdf_file.filename)
        pdf_file.save(input_path)

        # Ejecutar el script OCR con el archivo PDF
        try:
            subprocess.check_call([
                "python3", "ocr.py", input_folder, output_folder
            ])
        except subprocess.CalledProcessError as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

        # Leer el archivo de salida generado
        output_path = os.path.join(output_folder, pdf_file.filename.replace('.pdf', '.json'))
        if not os.path.exists(output_path):
            return jsonify({'error': 'Output file not generated'}), 500

        # Enviar el archivo de texto generado como respuesta
        # return send_file(output_path, as_attachment=True, download_name='output.txt', mimetype='text/plain')
        return send_file(output_path, as_attachment=True, download_name='output.json', mimetype='text/plain')

# Ejecutar el servidor
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
