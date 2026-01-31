Proyecto 7 – API de Predicción de Apps (Google Play)
Descripción
Este proyecto implementa un modelo de Machine Learning para predecir el segmento de una aplicación móvil (nicho, moderado, popular o viral) a partir de sus características.
El modelo es expuesto mediante una API REST desarrollada en Flask, accesible públicamente desde internet.
________________________________________
Modelo de Machine Learning
•	Tipo: Clasificación multiclase
•	Algoritmo: Random Forest (con pipeline de preprocesamiento)
•	Input: Variables numéricas y categóricas de apps de Google Play
•	Output:
o	Clase predicha
o	Probabilidades por clase
________________________________________
API REST
•	Framework: Flask
•	Endpoints principales:
o	/health → Verifica el estado de la API
o	/predict → Retorna la predicción del modelo a partir de un JSON de entrada
Ejemplo de respuesta
{
  "pred_clase": "popular",
  "prob_max": 0.68,
  "probs": {
    "nicho": 0.02,
    "moderado": 0.16,
    "popular": 0.68,
    "viral": 0.12
  }
}
________________________________________
Despliegue
•	La API se expone públicamente utilizando ngrok.
•	URL pública de ejemplo:
https://leverlike-solutus-alyse.ngrok-free.dev
________________________________________
Validación
•	La API fue validada enviando datos reales del dataset original.
•	El modelo devuelve predicciones coherentes y variables, confirmando el correcto funcionamiento del pipeline completo.
________________________________________
Tecnologías utilizadas
•	Python
•	pandas, scikit-learn
•	Flask
•	ngrok
________________________________________
Autor
Proyecto desarrollado por Edduardo Marchán, con fines académicos para el curso de Data Science / Machine Learning.
