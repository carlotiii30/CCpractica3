from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import when

# - - - - Crear sesión - - - - -
spark = SparkSession.builder.appName(
    "P3_ClasificacionEstelarExperimentos"
).getOrCreate()

# - - - - Cargar datos - - - - -
df = spark.read.csv(
    "hdfs://namenode:8020/user/carlota/half_celestial.csv",
    header=True,
    sep=";",
    inferSchema=True,
)

df = df.withColumn("class", when(df["type"] == "galaxy", 0.0).otherwise(1.0))
features = [
    "expAB_z",
    "i",
    "q_r",
    "modelFlux_r",
    "expAB_i",
    "expRad_u",
    "q_g",
    "psfMag_z",
    "dec",
    "psfMag_r",
]

# Preprocesamiento
assembler = VectorAssembler(inputCols=features, outputCol="features_vec")
scaler = StandardScaler(inputCol="features_vec", outputCol="features")

# División
(trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)

# Evaluador
evaluator = BinaryClassificationEvaluator(labelCol="class", metricName="areaUnderROC")

# Modelos y parametrizaciones
models = [
    (
        "LogisticRegression",
        LogisticRegression(labelCol="class", featuresCol="features", maxIter=10),
    ),
    (
        "LogisticRegression",
        LogisticRegression(labelCol="class", featuresCol="features", maxIter=50),
    ),
    (
        "RandomForest",
        RandomForestClassifier(labelCol="class", featuresCol="features", numTrees=10),
    ),
    (
        "RandomForest",
        RandomForestClassifier(labelCol="class", featuresCol="features", numTrees=50),
    ),
    (
        "GBTClassifier",
        GBTClassifier(labelCol="class", featuresCol="features", maxIter=10),
    ),
    (
        "GBTClassifier",
        GBTClassifier(labelCol="class", featuresCol="features", maxIter=50),
    ),
]

# Evaluación
results = []

for name, model in models:
    scaler = StandardScaler(inputCol="features_vec", outputCol="features")
    pipeline = Pipeline(stages=[assembler, scaler, model])

    try:
        pipeline_model = pipeline.fit(trainingData)
        predicciones = pipeline_model.transform(testData)
        auc = evaluator.evaluate(predicciones)
        results.append((name, str(model.extractParamMap()), round(auc, 4)))
    except Exception as e:
        print(f"Error con modelo {name}: {e}")
        results.append((name, str(model.extractParamMap()), None))


# Guardar resultados
results_df = spark.createDataFrame(results, ["Modelo", "Parametros", "AUC"])
results_df.coalesce(1).write.csv(
    "results_local/comparativa_modelos", header=True, mode="overwrite"
)
