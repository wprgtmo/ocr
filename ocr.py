from pdf2image import convert_from_path
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
from dotenv import load_dotenv
import numpy as np
import os
import cv2
import io
import google.generativeai as genai
import logging
import sys
import json

# Cargar variables de entorno
load_dotenv()
Image.MAX_IMAGE_PIXELS = None 
# Configuración del log
logging.basicConfig(
    filename="ocr_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
Image.MAX_IMAGE_PIXELS = None  
# Configura Google Vision y Gemini
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)

credentials = service_account.Credentials.from_service_account_file(
    "gen-lang-client-0296035343-be1037cd8bac.json"
)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

def cargar(file_path):
    """
    Convierte un archivo de imagen (JPG, PNG, PDF, TIFF) en una lista de objetos PIL.Image.

    Args:
        file_path (str): Ruta al archivo de entrada.

    Returns:
        list: Lista de imágenes PIL.Image.
    """
    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")

    pil_images = []
    file_extension = file_path.lower().split('.')[-1]

    try:
        if file_extension in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
            # Leer imágenes estándar
            image = Image.open(file_path)
            if image.format == "TIFF":
                # TIFF puede contener múltiples páginas
                for frame in range(image.n_frames):
                    image.seek(frame)
                    pil_images.append(image.copy())
            else:
                pil_images.append(image)

        elif file_extension == 'pdf':
            pil_images = convert_from_path(file_path, dpi=300) 
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_extension}")

    except Exception as e:
        print(f"Error al procesar el archivo {file_path}: {e}")

    return pil_images


def preprocess_images(pil_images):
    """
    Preprocesa una lista de imágenes en formato PIL.Image para optimizar el OCR.
    
    Args:
        pil_images (list of PIL.Image): Lista de imágenes PIL a procesar.
        
    Returns:
        list of PIL.Image: Lista de imágenes procesadas.
    """
    processed_images = []

    for pil_image in pil_images:
        # Convertir PIL.Image a formato numpy (OpenCV utiliza este formato)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Desenfoque: Aplicar filtro mediano para eliminar ruido
        image = cv2.medianBlur(image, 3)

        # 3. Binarización: Convertir a escala de grises y aplicar umbral
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar umbralización adaptativa
        processed1 = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Método de cálculo de umbral
            cv2.THRESH_BINARY,  # Tipo de binarización
            blockSize=55,  # Tamaño de la región para calcular el umbral
            C=15  # Constante para ajustar el umbral
        )

        processed = cv2.adaptiveThreshold(
            processed1, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Método de cálculo de umbral
            cv2.THRESH_BINARY,  # Tipo de binarización
            blockSize=35,  # Tamaño de la región para calcular el umbral
            C=7  # Constante para ajustar el umbral
        )

        target_width = 2100
        h, w = image.shape[:2]
        scale_ratio = target_width / w
        target_dimensions = (target_width, int(h * scale_ratio))
        processed = cv2.resize(processed, target_dimensions, interpolation=cv2.INTER_CUBIC)
        _, processed = cv2.threshold(processed, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convertir la imagen procesada de nuevo a formato PIL
        processed_pil_image = Image.fromarray(processed)
        processed_images.append(processed_pil_image)

    return processed_images


def obtener_datos_ocr(pil_images, lang="spa"):
    """
    Procesa una lista de imágenes con Google Vision para extraer texto.

    Args:
        pil_images (list of PIL.Image): Lista de imágenes en formato PIL.Image.
        api_key (str): (No utilizado en Google Vision).
        lang (str): Idioma esperado del texto OCR.

    Returns:
        list: Lista de listas, donde cada lista interna contiene líneas de palabras extraídas.
    """
    results = []

    for pil_image in pil_images:
        # Convertir PIL.Image a bytes
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        # Enviar la imagen a Google Vision
        image = vision.Image(content=image_bytes.getvalue())
        image_context = vision.ImageContext(language_hints=[lang])
        response = vision_client.text_detection(image=image, image_context=image_context)

        if response.error.message:
            logging.error(f"Error en Vision API: {response.error.message}")
            continue
        results.append(response.full_text_annotation.text)
    return results

def extract_text_from_list(resultado, output_folder, filename, api_key):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    result = ""
    
    with open(output_path, "w", encoding="utf-8") as f:
        for index,page in enumerate(resultado):
            if page:  # Guarda el texto si no está vacío
                result+=f"Page #{index}\n"
                result += page
                result += "\n\n\n"
        result = procesar_texto_geminis(api_key, result)
        f.write(result)



def  procesar_texto_geminis(api_key, text):
    """
    Procesa texto utilizando Gemini para correcciones y formato.

    Args:
        api_key (str): Clave API de Gemini.
        text (str): Texto extraído con OCR.

    Returns:
        str: Texto corregido.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Corrección profesional de texto extraído por OCR con los siguientes criterios:
    1. Corregir errores de reconocimiento óptico de caracteres
    2. Restaurar la puntuación y la estructura gramatical correcta
    3. Conservar el formato original del documento
    4. Mantener intacto el contenido informativo del texto original
    5. Asegurar la máxima precisión en la transcripción
    Características de salida:
    - Texto completamente legible
    - Sin caracteres extraviados o mal reconocidos
    - Gramática y ortografía  en otra carpeta.

    Args:corregidas
    - Formato apropiado para archivo .txt
    Texto OCR:
    {text}
    Texto corregido:
    """  
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=2000, temperature=0.2, top_p=0.75)
    )
    return response.text


def datos_json():
    data = {
        "patient": {
            "name": "",
            "dob": "",
            "address": "",
            "phone": "",
            "emergency_contact": {
                "name": "",
                "relation": "",
                "phone": ""
            }
        },
        "medical_information": {
            "weight": "",
            "height": "",
            "diagnoses": [],
            "medical_history": [],
            "surgical_history": [],
            "current_medications": [],
            "allergies": "",
            "vitals": {
                "pulse": "",
                "blood_pressure": "",
                "temperature": "",
                "respiratory_rate": ""
            }
        },
        "treatment": {
            "admission_date": "",
            "discharge_date": "",
            "treatment_plan": "",
            "dme": "",
            "follow_up": ""
        },
        "provider": {
            "name": "",
            "address": "",
            "contact": "",
            "signature_date": ""
        },
        "additional_notes": []
    }
    json_string = json.dumps(data, indent=4)
    return json_string


def  devolver_json_datos_paciente(api_key, text):
    """
    Procesa texto utilizando Gemini para obtener los datos del paciente

    Args:
        api_key (str): Clave API de Gemini.
        text (str): Texto extraído con OCR.

    Returns:
        str: Texto corregido.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Quiero que analices el siguiente texto: {text} \n y extraigas toda la información relacionada con un paciente en formato JSON. 
    La información debe incluir, cuando esté disponible:

    1. Información personal:
    - Nombre completo
    - Fecha de nacimiento (DOB)
    - Dirección
    - Teléfono de contacto
    - Contacto de emergencia (nombre, relación y teléfono)

    2. Información médica:
    - Peso
    - Altura
    - Diagnósticos
    - Historial médico y quirúrgico
    - Medicamentos actuales (dosis y frecuencia)
    - Alergias
    - Signos vitales (frecuencia cardíaca, presión arterial, temperatura, etc.)

    3. Información sobre el tratamiento:
    - Fecha de ingreso y alta
    - Plan de tratamiento o instrucciones de cuidado
    - Equipo médico requerido (DME)
    - Recomendaciones de seguimiento (F/U)

    4. Información del proveedor de salud:
    - Nombre del médico tratante
    - Dirección del consultorio
    - Número de contacto
    - Firma y fecha (si aplica)

    5. Datos adicionales:
    - Observaciones del personal médico
    - Historia social o familiar relevante
    - Resultados de exámenes relevantes

    Devuelve el resultado en este formato JSON estándar:

    ```json
    {datos_json()}
    ``` 
    """  
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=2000, temperature=0.2, top_p=0.75)
    )
    return response.text

def extract_json_from_text(resultado, output_folder, filename, api_key):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    result = ""
    
    with open(output_path, "w", encoding="utf-8") as f:        
        result = devolver_json_datos_paciente(api_key, resultado)
        f.write(result)

    

def process_folder(input_folder, output_folder, progress_hook=None):
    """
    Procesar todos los documentos de una carpeta y guardar los resultados
        input_folder (str): Carpeta de entrada con imágenes o documentos.
        output_folder (str): Carpeta de salida para guardar las imágenes procesadas.
        progress_hook (function): Función de progreso opcional.
    """
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Recorrer todos los archivos de la carpeta de entrada
    files = os.listdir(input_folder)
    for i, file_name in enumerate(files):
        input_path = os.path.join(input_folder, file_name)
        images_list = cargar(input_path)
        processed_images = preprocess_images(images_list)

        base_name = os.path.splitext(file_name)[0]

        results_api = obtener_datos_ocr(processed_images)
        extract_text_from_list(results_api, output_folder, f"{base_name}.txt", GENAI_API_KEY)
        extract_json_from_text(results_api, output_folder, f"{base_name}.json", GENAI_API_KEY)

        if progress_hook:
            progress_hook(i + 1)

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usa: python ocr.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    process_folder(input_folder, output_folder)