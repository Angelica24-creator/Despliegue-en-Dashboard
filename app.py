
# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import r2_score



# Configuración de página
st.set_page_config(page_title="Dashboard Airbnb Rio", layout="wide")

# Estilos personalizados con CSS
st.markdown(
    """
    <style>

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0daa5;
    }


    /* Texto del sidebar */
    [data-testid="stSidebar"] * {
        color: black;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Carga de datos
@st.cache_resource
def load_data():
    df = pd.read_csv("Rio de Janeiro sin atipicos.csv")
    df = df.dropna()  # Eliminamos registros vacíos
    return df

df = load_data()

# Título principal
st.title("🏡 Dashboard de Airbnb - Río de Janeiro")

# Selección de etapa en el sidebar
section = st.sidebar.radio("Selecciona la etapa:", ["Exploración de Datos", "Modelado Predictivo"])

# --- ETAPA 1: Exploración de datos ---


if section == "Exploración de Datos":
    st.header("Etapa I: Análisis Univariado")

    st.sidebar.subheader("Selecciona la variable a visualizar:")

    var = st.sidebar.selectbox("Variable categórica:", [
        "host_response_time", 
        "host_is_superhost",
        "host_verifications", 
        "host_has_profile_pic", 
        "host_identity_verified",
        "property_type", 
        "room_type", 
        "has_availability", 
        "instant_bookable"
    ])

    st.subheader(f"Distribución de {var}")

    if var in ["host_response_time", "host_is_superhost", "host_has_profile_pic",
               "host_identity_verified", "has_availability", "instant_bookable"]:
        fig = px.pie(df, names=var, title=f"Distribución de {var}", hole=0.3)
        fig.update_traces(textinfo='percent+label+value')
        st.plotly_chart(fig)

        if st.checkbox(f"Mostrar valores de '{var}'"):
            conteo = df[var].value_counts().reset_index()
            conteo.columns = [var, "count"]
            st.dataframe(conteo)

    elif var == "host_verifications":
        verifications = df["host_verifications"].dropna().str.replace("[\[\]']", '', regex=True).str.split(", ")
        all_verifications = verifications.explode().value_counts().reset_index()
        all_verifications.columns = ["Verificación", "Conteo"]
        fig = px.bar(all_verifications, x="Verificación", y="Conteo", text="Conteo", title="Verificaciones del host")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)

        if st.checkbox("Mostrar valores de 'host_verifications'"):
            st.dataframe(all_verifications)

    elif var in ["property_type", "room_type"]:
        conteo = df[var].value_counts().reset_index()
        conteo.columns = [var, "count"]
        fig = px.bar(conteo, x=var, y="count", text="count",
                     labels={var: var.capitalize(), "count": "Cantidad"},
                     title=f"Distribución de {var}")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)

        if st.checkbox(f"Mostrar valores de '{var}'"):
            st.dataframe(conteo)


# --- ETAPA 2: Modelado predictivo ---
elif section == "Modelado Predictivo":
    st.header("Etapa II: Modelado Predictivo")


    with st.sidebar:
        model_type = st.selectbox("Tipo de modelo:", 
                                  ["Regresión Lineal Simple", "Regresión Lineal Múltiple", "Regresión Logística"])

    if model_type == "Regresión Lineal Simple":
        st.subheader("Regresión Lineal Simple (predecir precio)")
        st.subheader("Mapa de precios en Río de Janeiro 🗺️")

        # Filtrar solo datos válidos para el mapa
        df_map = df[["latitude", "longitude", "price"]].dropna()

        # Slider para seleccionar rango de precios
        min_price = int(df_map["price"].min())
        max_price = int(df_map["price"].max())

        price_range = st.slider(
            "Selecciona el rango de precios",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )

        # Filtrar el DataFrame por el rango elegido
        df_map = df_map[(df_map["price"] >= price_range[0]) & (df_map["price"] <= price_range[1])]

        # Generar bins dinámicamente con 6 categorías
        step = (price_range[1] - price_range[0]) // 6 or 1  # evitar división por cero
        bins = list(range(price_range[0], price_range[1], step))
        if bins[-1] < price_range[1]:
            bins.append(price_range[1])  # asegurar que el último bin contenga el máximo

        labels = [f"{bins[i]} - {bins[i+1]-1}" for i in range(len(bins)-1)]

        # Crear columna de rangos
        df_map["price_range"] = pd.cut(df_map["price"], bins=bins, labels=labels, include_lowest=True)

        # Mostrar mapa con colores por rango de precios
        fig = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            color="price",
            size_max=10,
            zoom=10,
            mapbox_style="carto-positron",
            title="Distribución geográfica de precios"
        )

        st.plotly_chart(fig)



        # Variables predictoras disponibles
        features = [
            "host_total_listings_count", "accommodates", "bathrooms", "bedrooms", "beds",
            "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication", "review_scores_location",
            "review_scores_value", "property_type", "room_type", "has_availability", "instant_bookable"
        ]

        # Selección de variable predictora
        selected_feature = st.selectbox("Selecciona la variable predictora (X):", features)
        
        # Eliminamos filas nulas para esas variables
        df_model = df[["price", selected_feature]].dropna()

        # Codificamos variables categóricas si es necesario
        if df_model[selected_feature].dtype == 'object':
            df_model[selected_feature] = pd.factorize(df_model[selected_feature])[0]

        # Entrenamiento
        X = df_model[[selected_feature]]
        y = df_model["price"]

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Visualización
        fig = px.scatter(df_model, x=selected_feature, y="price", trendline="ols",
                        title=f"Regresión lineal: {selected_feature} vs Price")
        st.plotly_chart(fig)
        st.write(f"**Coeficiente R²:** {r2:.2f}")

        
    elif model_type == "Regresión Lineal Múltiple":
        st.subheader("Regresión Lineal Múltiple (predecir precio)")

        st.subheader("Predicción del precio a partir de múltiples variables")

                # Variables disponibles para X
        available_vars = [
            "host_total_listings_count", "accommodates", "bathrooms", "bedrooms", "beds",
            "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication", "review_scores_location",
            "review_scores_value", "availability_60"
        ]

        selected_vars = st.multiselect("Selecciona variables predictoras (X):", available_vars)

        # Validar selección
        if selected_vars:
            df_reg = df[["price"] + selected_vars].dropna()

            # Definir X y Y
            X = df_reg[selected_vars]
            y = df_reg["price"]

            # Modelo de regresión
            model = LinearRegression()
            model.fit(X, y)

            # Predicción y R^2
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            st.subheader("📈 Coeficiente de determinación (R²)")
            st.metric(label="R² del modelo", value=round(r2, 4))

            # Tabla de correlación con respecto a "price"
            st.subheader("📊 Correlación de cada variable con el precio")
            correlations = df_reg.corr(numeric_only=True)["price"].drop("price")
            st.dataframe(correlations.to_frame(name="Correlación"))

            # Mapa de calor (opcional)
            if st.checkbox("Mostrar mapa de calor de correlación completo"):
                st.subheader("🔍 Mapa de calor de todas las variables")
                plt.figure(figsize=(12, 8))
                sns.heatmap(df_reg.corr(numeric_only=True), annot=True, cmap="coolwarm")
                st.pyplot(plt.gcf())
                plt.clf()

            # Gráfica de dispersión 2D
            st.subheader("📌 Gráficas de dispersión")
            for var in selected_vars:
                fig = px.scatter(df_reg, x=var, y="price", trendline="ols",
                                title=f"Relación entre {var} y precio")
                st.plotly_chart(fig)
        else:
            st.warning("Selecciona al menos una variable para realizar la regresión.")
     

    elif model_type == "Regresión Logística":
        st.header("Etapa III: Regresión Logística - Predicción de Superhost")

        # Preprocesamiento
        df["host_is_superhost"] = df["host_is_superhost"].map({'t': 1, 'f': 0, 'SIN INFORMACION':1}).fillna(0)

        # Variables categóricas a convertir
        response_map = {
            "within an hour": 1,
            "within a few hours": 2,
            "within a day": 3,
            "a few days or more": 4,
            "SIN INFORMACION": 0
        }
        df["host_response_time"] = df["host_response_time"].map(response_map).fillna(0)
        df["host_has_profile_pic"] = df["host_has_profile_pic"].map({'t': 1, 'f': 0}).fillna(0)
        df["host_identity_verified"] = df["host_identity_verified"].map({'t': 1, 'f': 0}).fillna(0)

        # Conversión de columnas a numéricas
        columnas = [
            "host_acceptance_rate", "host_listings_count", "minimum_nights", "number_of_reviews",
            "review_scores_rating", "review_scores_checkin", "review_scores_communication", "reviews_per_month"
        ]
        for col in columnas:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Selección de variables predictoras
        predictoras = [
            "host_response_time", "host_has_profile_pic", "host_identity_verified",
            "host_acceptance_rate", "host_listings_count", "minimum_nights", "number_of_reviews",
            "review_scores_rating", "review_scores_checkin", "review_scores_communication", "reviews_per_month"
        ]

        # Separar variables
        X = df[predictoras]
        y = df["host_is_superhost"]

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Entrenar el modelo
        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(X_train, y_train)

        # Predicciones
        y_pred = modelo.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)

        # Matriz de confusión
        st.subheader("📌 Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No Superhost", "Superhost"], yticklabels=["No Superhost", "Superhost"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        # Tabla de métricas
        st.subheader("📈 Métricas del Modelo")
        metricas = pd.DataFrame({
            "Métrica": ["Exactitud (Accuracy)", "Precisión (Precision)", "Sensibilidad (Recall)"],
            "Valor": [acc, prec, rec]
        })
        st.dataframe(metricas)

        # Coeficientes del modelo
        st.subheader("🔍 Coeficientes del Modelo")
        coef_df = pd.DataFrame({
            "Variable": X.columns,
            "Coeficiente": modelo.coef_[0]
        }).sort_values(by="Coeficiente", key=abs, ascending=False)
        st.dataframe(coef_df)
