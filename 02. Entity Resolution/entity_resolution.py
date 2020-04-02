#entity_resolution.py
import re
import operator
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

@udf(returnType=FloatType())
def jaccardCoefficient(list1,list2):
    num = len(list(set(list1) & set(list2)))
    den = len(list(set(list1) | set(list2)))
    jaccard = float(num/den)
    return jaccard

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        # reading stopWords, Amazon_sample and Google_sample files
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = spark.read.parquet(dataFile1).cache()
        self.df2 = spark.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):
        # concatenating cols in df
        df.registerTempTable("df")
        concat_df = spark.sql("SELECT *, CONCAT_WS(' ',{col0},' ',{col1}) AS concat FROM df".format(col0=cols[0],col1=cols[1]))
        # Tokenizing the concatenated columns
        regex_tokenizer = RegexTokenizer().setInputCol("concat").setOutputCol("tokenized").setPattern("\W+").setToLowercase(True)
        token_df = regex_tokenizer.transform(concat_df)
        # Stop-words removal
        remover = StopWordsRemover(inputCol="tokenized", outputCol="joinKey", stopWords=list(self.stopWordsBC))
        swrem_df = remover.transform(token_df)
        return swrem_df.drop('concat','tokenized')


    def filtering(self, df1, df2):
        # filtering using joinKeys
        df1.registerTempTable("df1")
        df2.registerTempTable("df2")
        df1_new = spark.sql("SELECT id, joinKey, explode(joinKey) as joinKeyE FROM df1")
        df2_new = spark.sql("SELECT id, joinKey, explode(joinKey) as joinKeyE FROM df2")
        df1_new.registerTempTable("df1")
        df2_new.registerTempTable("df2")
        candDF = spark.sql("SELECT DISTINCT df1.id as id1, df1.joinKey as joinKey1, df2.id as id2, df2.joinKey as joinKey2 FROM df1,df2 WHERE df1.joinKeyE = df2.joinKeyE")
        return candDF


    def verification(self, candDF, threshold):
        # calculating jaccard coefficient
        resultDF = candDF.withColumn("jaccard", jaccardCoefficient(candDF["joinKey1"],candDF["joinKey2"]))
        # verification of pairs using jaccard similarity score
        resultDF = resultDF.filter(resultDF["jaccard"]>=threshold)
        return resultDF

    def evaluate(self, result, groundTruth):
        r_len = len(result)
        t_len = len(list(set(result) & set(groundTruth)))
        # calculating precision, recall, F-score
        prec = float(t_len/r_len)
        rec = float(t_len/len(groundTruth))
        if prec+rec != 0:
            f_score = (2*prec*rec) / (prec+rec)
        else:
            f_score = "Not Applicable"
        return (prec,rec,f_score)

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()))

        candDF = self.filtering(newDF1, newDF2)
        print ("After Filtering: %d pairs left" %(candDF.count()))

        resultDF = self.verification(candDF, threshold)
        print ("After Verification: %d similar pairs" %(resultDF.count()))

        return resultDF


    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    spark = SparkSession.builder.appName('example code').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext

    er = EntityResolution("Amazon_sample", "Google_sample", "stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = spark.read.parquet("Amazon_Google_perfectMapping_sample").rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))
