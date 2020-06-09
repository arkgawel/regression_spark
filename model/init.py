from pyspark import sql, SparkConf, SparkContext
from pyspark.sql.types import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd

conf = SparkConf().setAppName("Spark Regression")
conf = (conf.setMaster('local[*]'))
sc = SparkContext(conf=conf)
sqlContext = sql.SQLContext(sc)


fields = [StructField('no', IntegerType(), True), StructField('cars', LongType(), True),
          StructField('price', FloatType(), True), StructField('population', FloatType(), True),
          StructField('avg_salary', FloatType(), True), StructField('avg_park_price', FloatType(), True)]
schema = StructType(fields)


#pyspark dataframe
df = sqlContext.read.load('./data.csv', format="csv", header="true", sep=";", schema=schema)
#df.show()
#df.printSchema()

#convert to pandas dataframe
#columns = ['no', 'cars', 'price', 'population','avg_salary','avg_park_price']
#df =pd.DataFrame(df, columns=columns)
#print(df)


#convert data to rdd
rdd = df.rdd.map(list).collect()
rdd = sc.parallelize(rdd)
print(rdd.take(2))

print('RDD Count:', rdd.count())
print('RDD Count by key:', rdd.countByKey())
print('RDD Num Partitions:', rdd.getNumPartitions())
#end#############################################################


#Data preparation into VectorAssembler
assembler = VectorAssembler(inputCols=['cars',
                                     'price',
                                     'population',
                                     'avg_salary']
                                     ,outputCol='features')

output = assembler.transform(df)
#output.select('features','avg_park_price').show(5)
output.select('features','avg_park_price')

split_data = output.randomSplit([0.6,0.4])
train_data, test_data = split_data
train_data.describe().show()


#Building model
linear_model = LinearRegression(featuresCol='features',labelCol='avg_park_price')
trained_model = linear_model.fit(train_data)
results = trained_model.evaluate(train_data)

print('R^2 Error :',results.r2)

#Making predictions
new_data = test_data.select('features')
predictions = trained_model.transform(new_data)
predictions.show()

#Converiting to Pandas DF
predictions = predictions.toPandas()
test_data = test_data.toPandas()


#Making visualization
plt.scatter(test_data['avg_park_price'], predictions['prediction'] ,  color='black')
plt.xlabel('Actual - avg_park_price')
plt.ylabel('Predicted - avg_park_price')
plt.title('Actual vs. Predicted')
plt.show()
