#Initialize SparkSession and SparkContext
from pyspark.sql import SparkSession

#Create a Spark Session
SpSession = SparkSession \
    .builder \
    .master("local") \
    .appName("py_spark") \
    .getOrCreate()

SpContext = SpSession.sparkContext

"""--------------------------------------------------------------------------
Load Data
-------------------------------------------------------------------------"""

#Load the CSV file into a RDD
dataLines = SpContext.textFile("leno_jeeva/classification.csv",)
dataLines.cache()
dataLines.count()
dataLines.take(10)
"""--------------------------------------------------------------------------
Cleanup Data
-------------------------------------------------------------------------"""

from pyspark.sql import Row
#Create a Data Frame from the data
parts = dataLines.map(lambda l: l.split(","))
parts.take(5)
churnMap = parts.map(lambda p: Row(c_id=float(p[0]), \
                                  gender=float(p[1]), \
                                  seniorcitizen=float(p[2]), \
                                  partner=float(p[3]), \
                                  dependants=float(p[4]), \
                                  tenure=float(p[5]), \
                                  phone_service=float(p[6]), \
                                  multiple_lines=float(p[7]), \
                                  internetservice=float(p[8]), \
                                  online_security=float(p[9]), \
                                  online_backup=float(p[10]), \
                                  device_protection=float(p[11]), \
                                  tech_support=float(p[12]), \
                                  streaming_movies=float(p[13]), \
                                  streaming_TV=float(p[14]), \
                                  contract=float(p[15]), \
                                  paperlessbilling=float(p[16]), \
                                  payment_method=float(p[17]), \
                                  monthly_charges=float(p[18]), \
                                  total_charges=float(p[19]), \
                                  churn=p[20]))
churnMap.take(5)                             
# Infer the schema, and register the DataFrame as a table.
churndf = SpSession.createDataFrame(churnMap)
churndf.cache()

#Add a numeric indexer for the label/target column
from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol="churn", outputCol="ind_churn")
si_model = stringIndexer.fit(churndf)
churnDf = si_model.transform(churndf)

churnDf.select("churn","ind_churn").distinct().show()
churnDf.cache()

"""--------------------------------------------------------------------------
Perform Data Analytics
-------------------------------------------------------------------------"""

#See standard parameters
churnDf.describe().show()

#Find correlation between predictors and target
for i in churnDf.columns:
    if not(isinstance(churnDf.select(i).take(1)[0][0], unicode)) :
        print("Correlation to Churn for ", i, \
                    churnDf.stat.corr('ind_churn',i))

"""--------------------------------------------------------------------------
Prepare data for ML
-------------------------------------------------------------------------"""

#Transform to a Data Frame for input to Machine Learing
#Drop columns that are not required (low correlation)

from pyspark.ml.linalg import Vectors
def transformToLabeledPoint(row) :
    lp = ( row["churn"], row["ind_churn"], \
                Vectors.dense([row["contract"],\
                        row["tenure"], \
                        row["tech_support"], \
                        row["online_security"], \
                        row["paperlessbilling"], \
                        row["online_backup"], \
                        row["monthly_charges"], \
                        row["device_protection"]]))
    return lp
    
churnLp = churnDf.rdd.map(transformToLabeledPoint)
churnLpDf = SpSession.createDataFrame(churnLp,["churn","label", "features"])
churnLpDf.select("churn","label","features").show(10)
churnLpDf.cache()

"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""
#Split into training and testing data
(trainingData, testData) = churnLpDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

#Create the model
from pyspark.ml.classification import DecisionTreeClassifier
dtClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="label",\
                featuresCol="features")
dtModel = dtClassifer.fit(trainingData)

dtModel.numNodes
dtModel.depth

#Predict on the test data
predictions = dtModel.transform(testData)
predictions.select("prediction","churn","label").show()

#Evaluate accuracy
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
print("Accuracy: " + str(evaluator.evaluate(predictions)))

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="weightedPrecision")
print("Precision: " + str(evaluator.evaluate(predictions)))

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="weightedRecall")
print("Recall: " + str(evaluator.evaluate(predictions)))

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="f1")
print("F1: " + str(evaluator.evaluate(predictions)))

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()
