import os
import getpass
import pickle
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, send_file
from garminconnect import Garmin

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# -----------------------
# Funciones auxiliares
# -----------------------

def extraer_datos(api, dias):
    """Descarga actividades y construye dataframe de laps"""
    actividades = api.get_activities(0, 1200)
    df = pd.DataFrame(actividades)

    # Filtro temporal
    df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"])
    fecha_max = df["startTimeLocal"].max()
    fecha_min = fecha_max - pd.Timedelta(days=dias)
    df_filtrado = df[df["startTimeLocal"] >= fecha_min].copy()

    if df_filtrado.empty:
        return pd.DataFrame(), 0

    # Ejemplo simple: usar distancias y tiempos
    df_filtrado["activityId"] = df_filtrado["activityId"].astype(str)
    df_filtrado["dist_cum"] = df_filtrado["distance"]  # metros
    df_filtrado["dur_min_cum"] = df_filtrado["duration"] / 60  # minutos
    df_filtrado["ritmo_lap"] = df_filtrado["duration"] / df_filtrado["distance"]

    return df_filtrado, df_filtrado["activityId"].nunique()


def entrenar_modelo(df):
    """Entrena regresión lineal"""
    X = df[["dist_cum", "dur_min_cum"]]
    y = df["ritmo_lap"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, r2


# -----------------------
# Flask
# -----------------------
app = Flask(__name__)
mejor_modelo = None
mejor_r2 = -999
archivo_modelo = None


@app.route("/", methods=["GET", "POST"])
def index():
    global mejor_modelo, mejor_r2, archivo_modelo

    if request.method == "POST":
        usuario = request.form["usuario"]
        password = request.form["password"]

        # Autenticación Garmin
        try:
            api = Garmin(usuario, password)
            api.login()
        except Exception as e:
            return f"❌ Error en login Garmin: {e}"

        # Rango de días a evaluar
        dias_range = [30, 60, 90, 180]
        min_actividades = 20
        mejor_modelo, mejor_r2 = None, -999

        for dias in dias_range:
            df_laps, n_actividades = extraer_datos(api, dias)
            if n_actividades < min_actividades:
                continue

            modelo, r2 = entrenar_modelo(df_laps)

            if r2 > mejor_r2:
                mejor_modelo = modelo
                mejor_r2 = r2

        if mejor_modelo is None:
            return "❌ No se encontró un modelo válido (pocas actividades)."

        # Guardar modelo en archivo
        fecha = datetime.now().strftime("%d%m%Y")
        archivo_modelo = f"{usuario}_{fecha}.pkl"
        with open(archivo_modelo, "wb") as f:
            pickle.dump(mejor_modelo, f)

        return render_template("resultado.html", r2=mejor_r2, archivo=archivo_modelo)

    return render_template("index.html")


@app.route("/download")
def download():
    global archivo_modelo
    if archivo_modelo and os.path.exists(archivo_modelo):
        return send_file(archivo_modelo, as_attachment=True)
    return "❌ No hay modelo disponible para descargar."


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
