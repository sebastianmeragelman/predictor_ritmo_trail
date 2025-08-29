import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from garminconnect import Garmin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from tqdm import tqdm

st.title("Análisis Garmin: Modelo de Ritmo por Lap")

# ----------------------------
# Paso 1: Credenciales
# ----------------------------
usuario = st.text_input("Usuario Garmin")
password = st.text_input("Contraseña Garmin", type="password")

if st.button("Conectar y Procesar"):

    if not usuario or not password:
        st.error("Debe ingresar usuario y contraseña.")
    else:
        try:
            api = Garmin(usuario, password)
            api.login()
            st.success("✅ Conexión exitosa a Garmin Connect")
        except Exception as e:
            st.error(f"Error en login Garmin: {e}")
            st.stop()

        # ----------------------------
        # Paso 2: Descargar actividades
        # ----------------------------
        actividades = api.get_activities(0, 1200)
        df_act = pd.DataFrame(actividades)
        st.write(f"Se descargaron {len(df_act)} actividades.")

        # ----------------------------
        # Paso 3: Funciones de tu código
        # ----------------------------
        def extraer_datos(metros, cantidad_dias):
            df_act["startTimeLocal"] = pd.to_datetime(df_act["startTimeLocal"])
            df_filtrado = df_act[df_act["distance"] > metros].copy()
            hoy = pd.Timestamp.today()
            df_filtrado = df_filtrado[(hoy - df_filtrado["startTimeLocal"]).dt.days <= cantidad_dias].copy()

            laps_data = []
            for _, act in tqdm(df_filtrado.iterrows(), total=len(df_filtrado)):
                activity_id = act["activityId"]
                days_since_first = (act["startTimeLocal"] - df_act["startTimeLocal"].min()).days
                laps = api.get_activity_splits(activity_id)

                if "lapDTOs" in laps:
                    for lap in laps["lapDTOs"]:
                        dist_km = lap.get("distance", 0) / 1000
                        dur_min = lap.get("duration", 0) / 60
                        gain = lap.get("elevationGain", 0)
                        loss = lap.get("elevationLoss", 0)

                        if dist_km > 0:
                            laps_data.append({
                                "activityId": activity_id,
                                "lapNumber": lap.get("lapIndex"),
                                "dist_km": dist_km,
                                "dur_min": dur_min,
                                "ritmo_lap": dur_min / dist_km,
                                "alt_gain": gain,
                                "alt_loss": loss,
                                "pendiente_media": (gain - loss) / dist_km,
                                "days_since_first": days_since_first
                            })

            df_laps = pd.DataFrame(laps_data)
            df_laps = df_laps.sort_values(["activityId", "lapNumber"])
            df_laps["alt_gain_cum"] = df_laps.groupby("activityId")["alt_gain"].cumsum()
            df_laps["alt_loss_cum"] = df_laps.groupby("activityId")["alt_loss"].cumsum()
            df_laps["dist_cum"] = df_laps.groupby("activityId")["dist_km"].cumsum()
            df_laps["dur_min_cum"] = df_laps.groupby("activityId")["dur_min"].cumsum()
            return df_laps, df_filtrado["startTimeLocal"].min()

        def regresion_lineal(df_laps):
            X = df_laps[[
                "lapNumber", "dist_cum", "alt_gain_cum", "alt_loss_cum",
                "alt_loss", "alt_gain", "dur_min_cum", "days_since_first"
            ]]
            y = df_laps["ritmo_lap"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"R²: {r2:.3f} | MAE: {mae:.3f}")
            return model

        def test_r2_por_antiguedad(metros, dias_range, min_actividades=20):
            r2_list, laps_list, dias_validos, actividades_list = [], [], [], []
            mejor_modelo, mejor_r2 = None, -999

            for dias in dias_range:
                df_laps, _ = extraer_datos(metros, dias)
                n_actividades = df_laps["activityId"].nunique()
                if n_actividades < min_actividades:
                    continue
                model = regresion_lineal(df_laps)
                X = df_laps[[
                    "lapNumber", "dist_cum", "alt_gain_cum", "alt_loss_cum",
                    "alt_loss", "alt_gain", "dur_min_cum", "days_since_first"
                ]]
                y = df_laps["ritmo_lap"]
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                dias_validos.append(dias)
                r2_list.append(r2)
                laps_list.append(len(df_laps))
                actividades_list.append(n_actividades)
                if r2 > mejor_r2:
                    mejor_r2 = r2
                    mejor_modelo = model

            return dias_validos, r2_list, laps_list, actividades_list, mejor_modelo, mejor_r2

        # ----------------------------
        # Paso 4: Ejecutar análisis
        # ----------------------------
        dias_range = list(range(100, 500, 30))
        metros = 20000
        dias_validos, r2_list, laps_list, actividades_list, mejor_modelo, mejor_r2 = test_r2_por_antiguedad(metros, dias_range)
        st.write(f"Mejor R²: {mejor_r2:.3f}")

        # ----------------------------
        # Paso 5: Gráfico
        # ----------------------------
        fig, ax1 = plt.subplots(figsize=(9,5))
        ax1.plot(dias_validos, r2_list, marker='o', color="tab:blue", label="R²")
        ax1.set_xlabel("Antigüedad máxima (días)")
        ax1.set_ylabel("R²", color="tab:blue")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax2 = ax1.twinx()
        ax2.plot(dias_validos, laps_list, marker='s', color="tab:green", label="Laps usados")
        ax2.plot(dias_validos, actividades_list, marker='^', color="tab:red", label="Actividades")
        ax2.set_ylabel("Cantidad de laps / actividades", color="tab:green")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        st.pyplot(fig)

        # ----------------------------
        # Paso 6: Guardar y descargar modelo
        # ----------------------------
        fecha = datetime.now().strftime("%d%m%Y")
        nombre_archivo = f"{usuario}_{fecha}_mejor_modelo.pkl"
        joblib.dump(mejor_modelo, nombre_archivo)
        st.success(f"✅ Modelo guardado: {nombre_archivo}")
        st.download_button("📥 Descargar modelo", data=open(nombre_archivo, "rb"), file_name=nombre_archivo)
