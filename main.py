from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import comet_ml
import huggingface_hub
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import uvicorn

# Charger les variables d'environnement
load_dotenv()

app = FastAPI()

# Configuration
WORKSPACE = "loule95450"
PROJECT_NAME = "face-classification"
REPO_ID = "Loule/face-classification"
MODEL_FILENAME = "fine_tuned_face_classifier_model.keras"
PORT = int(os.getenv('PORT', 8000))
last_model_check = None
current_model_path = None
model = None

# Initialisation des clients
comet_ml.login(
    project_name=PROJECT_NAME, 
    workspace=WORKSPACE, 
    api_key=os.getenv('COMET_API_KEY')
)
api = comet_ml.API()
huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))

def get_latest_model():
    experiences = api.get_experiments(WORKSPACE, PROJECT_NAME)
    experiences.reverse()
    
    for experience in experiences:
        try:
            files_in_branch = huggingface_hub.list_repo_files(repo_id=REPO_ID, revision=experience.id)
            if MODEL_FILENAME in files_in_branch:
                return experience.id
        except:
            continue
    return None

def load_model_if_needed():
    global model, last_model_check, current_model_path
    
    # Vérifier le modèle toutes les 6 heures
    if (last_model_check is None or 
        datetime.now() - last_model_check > timedelta(hours=6)):
        
        experiment_id = get_latest_model()
        if experiment_id:
            new_model_path = huggingface_hub.hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILENAME,
                revision=experiment_id
            )
            
            if new_model_path != current_model_path:
                model = tf.keras.models.load_model(new_model_path)
                current_model_path = new_model_path
        
        last_model_check = datetime.now()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Fonction pour prétraiter l'image
def preprocess_image(image):
    image = image.resize((128, 128))  # Redimensionne l'image à 128x128 pixels
    image_array = np.array(image)  # Convertit l'image en tableau numpy
    return np.expand_dims(image_array, axis=0)  # Ajoute une dimension pour la batch size


# Route de prédiction d'image
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Vérifier et charger le dernier modèle si nécessaire
    load_model_if_needed()
    
    if model is None:
        return {"error": "No model available"}
    
    # Lire et prétraiter l'image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    processed_image = preprocess_image(image)
    
    # Faire la prédiction
    prediction = model.predict(processed_image)
    fake_probability = float(prediction[0][0])
    
    return {
        "filename": file.filename,
        "fake_probability": fake_probability,
        "is_fake": fake_probability > 0.5,
        "model_used": MODEL_FILENAME
    }

@app.get("/")
async def root():
    return {"message": "Image Detection API is running"}

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)


@app.get("/healthcheck")
async def healthcheck():
    # Vérification du modèle
    model_status = "loaded" if model is not None else "not loaded"
    
    # Vérification des connexions externes (optionnel)
    try:
        comet_connected = True
        huggingface_connected = True
        # Exemple : vérification d'une connexion rapide aux services
        api.get_account_details()
        huggingface_hub.list_repo_files(repo_id=REPO_ID)
    except:
        comet_connected = False
        huggingface_connected = False
    
    # Retourner l'état général
    return {
        "status": "ok" if model_status == "loaded" and comet_connected and huggingface_connected else "error",
        "details": {
            "model": model_status,
            "comet_ml_connection": "ok" if comet_connected else "error",
            "huggingface_connection": "ok" if huggingface_connected else "error",
        }
    }
