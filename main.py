from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

# Charge le modèle sans l'argument custom_objects
model = tf.keras.models.load_model('model.keras')

# Ajoute le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Changez ici si vous voulez restreindre à des origines spécifiques
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
    image_data = await file.read()  # Lit l'image téléchargée
    image = Image.open(io.BytesIO(image_data))  # Ouvre l'image
    processed_image = preprocess_image(image)  # Prétraite l'image
    
    # Effectue la prédiction
    prediction = model.predict(processed_image)
    fake_probability = float(prediction[0][0])  # Supposons que le modèle retourne une probabilité unique
    
    return {
        "filename": file.filename,
        "fake_probability": fake_probability,
        "is_fake": fake_probability > 0.5  # Détermine si l'image est considérée comme fake
    }

# Route racine pour vérifier si l'API fonctionne
@app.get("/")
async def root():
    return {"message": "Image Detection API is running"}
