from flask import Flask, request
from flask.templating import render_template
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import sklearn
from sklearn.pipeline import make_pipeline 
# skimage
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io

App= Flask(__name__, template_folder="Plantillas")
BASE_PATH= os.getcwd()
UPLOAD_PATH= os.path.join(BASE_PATH, "static", "subidas")
MODEL_PATH= os.path.join(BASE_PATH, "static", "models")
#------------------------Cargar Modelos----------------------------------
model_sgd_path= os.path.join(MODEL_PATH, "Clasificación_de_Imagenes.pickle")
scaler_path= os.path.join(MODEL_PATH, "Escalar.pickle")
model_sgd= pickle.load(open(model_sgd_path, "rb"))
scaler= pickle.load(open(scaler_path, "rb"))

@App.errorhandler(404)
def error404(error):
    mensaje= "Ah Ocurrido Un Error 404. Pagina no encontrada. Por Favor regresa al sitio de inicio e intenta de nuevo"
    return render_template("Error.html", message=mensaje) #Pagina no Encontrada

@App.errorhandler(405)
def error405(error):
    mensaje= "Error 405. Metodo no encontrado"
    return render_template("Error.html", message=mensaje)

@App.errorhandler(500)
def error500(error):
    mensaje= "Error Interno 500. Error ocurrido en el programa"
    return render_template("Error.html", message=mensaje)

@App.route("/", methods=["GET", "POST"])
def inicio():
    if request.method== "POST":
        subir_archivo= request.files["image_name"]
        nombre_archivo= subir_archivo.filename 
        print("El nombre del archivo que ah subido es =", nombre_archivo)
        # Reconociendo la extensión del archivo, solo .png .jpg .jpeg
        extension= nombre_archivo.split(".")[-1] #Esto obtiene lo ultimo del archivo lo que le sigue al punto
        print("La extensión del archivo es =", extension)
        if extension.lower() in ["png", "jpg", "jpeg"]:
            #Guardando las imágenes
            path_save= os.path.join(UPLOAD_PATH, nombre_archivo)
            subir_archivo.save(path_save)
            print("Archivo Subido Correctamente")
            # Enviando al pipeline Model
            resultados = pipeline_model(path_save, scaler, model_sgd)
            hei = getheight(path_save)
            print(resultados)
            return render_template("Subir.html", fileupload=True, extension=False, data=resultados, image_filename=nombre_archivo, height=hei)
        else:
            print("Usa solo los tipos de archivo permitido .png .jpg .jpeg")
            return render_template("Subir.html", extension=True, fileupload=False)
    else:
        return render_template("Subir.html", extension=False, fileupload=False)

@App.route("/acerca/")
def acerca():
    return render_template("Acerca.html")

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ = img.shape
    ascept= h/w
    given_height= 300
    height= given_height*ascept
    return height

def pipeline_model(path, Escalador_transform, Modelo_sgd):
    # pipeline model
    imagen= skimage.io.imread(path)
    #transformar imagen a 80x80 ya que asi se entreno el modelo
    imagen_resize= skimage.transform.resize(imagen, (80,80))
    imagen_scale= 255*imagen_resize
    imagen_transform= imagen_scale.astype(np.uint8)
    # rgb a gris
    imagen_gris=skimage.color.rgb2gray(imagen_transform)
    #hog feature
    feature_vector= skimage.feature.hog(imagen_gris, orientations=10, pixels_per_cell=(8,8), cells_per_block=(2, 2))
    # Scaling
    EscaladorX= Escalador_transform.transform(feature_vector.reshape(1, -1))
    Resultados=Modelo_sgd.predict(EscaladorX)
    #Función de Desicion #Confidence
    valor_decisións= Modelo_sgd.decision_function(EscaladorX).flatten()
    Etiquetas= Modelo_sgd.classes_
    #Z probability
    z =scipy.stats.zscore(valor_decisións)
    prob_valor= scipy.special.softmax(z)
    # Top 5 valores de probabilidad
    top_5_prob_ind=prob_valor.argsort()[::-1][:5]
    Top_Labs=Etiquetas[top_5_prob_ind]
    Top_Prob= prob_valor[top_5_prob_ind]
    #Poner en el diccionario
    Top_dict= dict()
    for key, val in zip(Top_Labs, Top_Prob):
        Top_dict.update({key:np.round(val,3)})
    return Top_dict

#if __name__ == "__main__":
    #App.run(debug=True) #debug cuando estamos probando la paguina no usar en produccion solo desarrollo