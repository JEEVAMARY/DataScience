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
GRData = SpContext.textFile("leno_jeeva/google_review.csv")
GRData.cache()
GRData.take(5)

"""--------------------------------------------------------------------------
Cleanup Data
-------------------------------------------------------------------------"""

from pyspark.sql import Row

#Function to cleanup Data
def CleanupData(inputStr) :
    attList=inputStr.split(",")
    
    #Create a row with cleaned up and converted data
    values= Row(    AVERAGE_RATINGS_CHURCHES=float(attList[0]),\
                    AVERAGE_RATINGS_RESORTS=float(attList[1]), \
                    AVERAGE_RATINGS_BEACHES=float(attList[2]), \
                    AVERAGE_RATINGS_PARKS=float(attList[3]), \
                    AVERAGE_RATINGS_THEATRES=float(attList[4]), \
                    AVERAGE_RATINGS_MUSEUMS=float(attList[5]), \
                    AVERAGE_RATINGS_MALLS=float(attList[6]), \
                    AVERAGE_RATINGS_ZOO=float(attList[7]), \
                    AVERAGE_RATINGS_RESTAURANTS=float(attList[8]), \
                    AVERAGE_RATINGS_PUBS_BARS=float(attList[9]), \
                    AVERAGE_RATINGS_LOCALSERVICES=float(attList[10]), \
                    AVERAGE_RATINGS_BURGER_PIZZASHOPS=float(attList[11]), \
                    AVERAGE_RATINGS_HOTELS_OTHER_LODGINGS=float(attList[12]), \
                    AVERAGE_RATINGS_JUICE_BARS=float(attList[13]), \
                    AVERAGE_RATINGS_ART_GALLERIES=float(attList[14]), \
                    AVERAGE_RATINGS_DANCE_CLUBS=float(attList[15]), \
                    AVERAGE_RATINGS_SWIMMINGPOOLS=float(attList[16]), \
                    AVERAGE_RATINGS_GYMS=float(attList[17]), \
                    AVERAGE_RATINGS_BAKERIES=float(attList[18]), \
                    AVERAGE_RATINGS_BEAUTY_SPAS=float(attList[19]), \
                    AVERAGE_RATINGS_CAFES=float(attList[20]), \
                    AVERAGE_RATINGS_VIEWPOINTS=float(attList[21]), \
                    AVERAGE_RATINGS_GARDENS=float(attList[22]), \
                    AVERAGE_RATINGS_MONUMENTS=float(attList[23]), \
                )
    
    return values

#Remove the first line (contains headers)
dataLines = GRData.filter(lambda x: "AVERAGE_RATINGS_MONUMENTS" not in x)
dataLines.count()

#Run map for cleanup
GRMap = dataLines.map(CleanupData)
GRMap.cache()
GRMap.take(5)

#Create a Data Frame with the data. 
GRDf = SpSession.createDataFrame(GRMap)
GRDf.describe().show()

"""--------------------------------------------------------------------------
Perform Data Analytics
-------------------------------------------------------------------------"""
#Find correlation between predictors and target
for i in GRDf.columns:
    if not( isinstance(GRDf.select(i).take(1)[0][0], unicode)) :
        print( "Correlation to Average Ratings of Monuments for ", i, GRDf.stat.corr('AVERAGE_RATINGS_MONUMENTS',i))


"""--------------------------------------------------------------------------
Prepare data for ML
-------------------------------------------------------------------------"""

#Transform to a Data Frame for input to Machine Learing
#Drop columns that are not required (low correlation)

from pyspark.ml.linalg import Vectors
def transformToLabeledPoint(row) :
    lp = ( row["AVERAGE_RATINGS_MONUMENTS"], Vectors.dense([row["AVERAGE_RATINGS_CHURCHES"],\
                        row["AVERAGE_RATINGS_RESORTS"], \
                        row["AVERAGE_RATINGS_BEACHES"], \
                        row["AVERAGE_RATINGS_PARKS"], \
                        row["AVERAGE_RATINGS_THEATRES"], \
                        row["AVERAGE_RATINGS_MUSEUMS"], \
                        row["AVERAGE_RATINGS_MALLS"], \
                        row["AVERAGE_RATINGS_ZOO"], \
                        row["AVERAGE_RATINGS_RESTAURANTS"], \
                        row["AVERAGE_RATINGS_PUBS_BARS"], \
                        row["AVERAGE_RATINGS_LOCALSERVICES"], \
                        row["AVERAGE_RATINGS_BURGER_PIZZASHOPS"], \
                        row["AVERAGE_RATINGS_HOTELS_OTHER_LODGINGS"], \
                        row["AVERAGE_RATINGS_JUICE_BARS"], \
                        row["AVERAGE_RATINGS_ART_GALLERIES"], \
                        row["AVERAGE_RATINGS_DANCE_CLUBS"], \
                        row["AVERAGE_RATINGS_SWIMMINGPOOLS"], \
                        row["AVERAGE_RATINGS_GYMS"], \
                        row["AVERAGE_RATINGS_BAKERIES"], \
                        row["AVERAGE_RATINGS_BEAUTY_SPAS"], \
                        row["AVERAGE_RATINGS_CAFES"], \
                        row["AVERAGE_RATINGS_VIEWPOINTS"], \
                        row["AVERAGE_RATINGS_GARDENS"]]))
    return lp
    
GRLp = GRMap.map(transformToLabeledPoint)
GRDF1 = SpSession.createDataFrame(GRLp,["label", "features"])
GRDF1.select("label","features").show(10)


"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""

#Split into training and testing data
(trainingData, testData) = GRDF1.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()

#Build the model on training data
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10)
lrModel = lr.fit(trainingData)

#Print the metrics
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

#Predict on the test data
predictions = lrModel.transform(testData)
predictions.select("prediction","label","features").show()

#Find R2 for Linear Regression
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="mse")
evaluator.evaluate(predictions)
print("MSE: " + str(evaluator.evaluate(predictions)))
