import mlflow

# 1. Configurar la dirección del servidor (apuntando al puerto 5003)
mlflow.set_tracking_uri("http://localhost:5003")

# 2. Crear y/o seleccionar el experimento
# Si el experimento "gestor-textos-similitud" no existe, MLflow lo crea automáticamente.
# Si ya existe, simplemente lo selecciona para registrar las siguientes ejecuciones.
mlflow.set_experiment("gestor-textos-similitud")

# 3. Cada vez que calcules la similitud, abres un "run" dentro de ese experimento
def registrar_busqueda_mlflow(texto_query, modelo, resultados):
    with mlflow.start_run():
        # Aquí registras tus parámetros y métricas
        mlflow.log_param("modelo", modelo)
        mlflow.log_param("query", texto_query)
        
        if resultados:
            mlflow.log_metric("max_similitud", resultados[0]['score'])