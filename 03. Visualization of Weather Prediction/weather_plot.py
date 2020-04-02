import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

import numpy as np
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession, functions, types
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName('weather map').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3' # make sure we have Spark 2.3+

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import elevation_grid as eg

def schema():
    tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),])
    return tmax_schema

def plot_map1(df,col_name,img_name, title,label='avg temperature in degree celsius', cmap='autumn_r'):
    plot = df.toPandas()
    lats = plot["latitude"].values
    lons = plot["longitude"].values
    temp = plot[col_name].values

    fig = plt.figure(img_name)

    m = Basemap(projection = 'merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='i')
    m.bluemarble(scale=0.2)
    m.drawcountries(color='black', linewidth=0.2)
    m.drawcoastlines(color='black', linewidth=0.2)
    m.drawmapboundary(color='black', linewidth=0.8)

    x, y = m(lons,lats)
    plt.scatter(x, y, c=temp, s=2, marker='o',cmap=cmap)
    plt.colorbar(label=label,orientation='horizontal')
    plt.title(title)
    plt.savefig(img_name+".png",dpi=600)

def plot_map2(df,col_name,img_name,date):
    plot = df.toPandas()
    lats = plot["latitude"].values
    lons = plot["longitude"].values
    temp = plot["prediction"].values

    fig = plt.figure(img_name)

    m = Basemap(projection='robin',lon_0=0,resolution='i')
    m.drawcountries(color='black', linewidth=0.2)
    m.drawcoastlines(color='black', linewidth=0.5)
    m.drawmapboundary(color='black', linewidth=0.5)

    x, y = m(lons,lats)
    plt.scatter(x, y, c=temp, s=2, marker='o', cmap='Paired_r')
    plt.colorbar(label='temperature in degree celsius',orientation='horizontal')
    plt.title("Predicted temperature map for "+str(date))
    plt.savefig(img_name+".png",dpi=600)

def main(inputs,model_file,test_file):

    ################################### Question 2-a
    data = spark.read.csv(inputs, schema=schema())
    data.registerTempTable("data")
    df = spark.sql("SELECT station, YEAR(date) as year, latitude, longitude, elevation, tmax FROM data").cache()
    df.registerTempTable("df")
    df1 = spark.sql("SELECT station,latitude,longitude,ROUND(AVG(tmax),4) as avg_temp FROM df WHERE year >= 1990 AND year < 2020 GROUP BY station, longitude, latitude")
    # df1 = normalize(df1)
    df2 = spark.sql("SELECT station,latitude,longitude,ROUND(AVG(tmax),4) as avg_temp FROM df WHERE year >= 1960 AND year < 1990 GROUP BY station, longitude, latitude")
    # df2 = normalize(df2)
    df3 = spark.sql("SELECT station,latitude,longitude,ROUND(AVG(tmax),4) as avg_temp FROM df WHERE year >= 1930 AND year < 1960 GROUP BY station, longitude, latitude")
    # df3 = normalize(df3)

    plot_map1(df1,"avg_temp","1990-2020",title = 'Average temperature of stations in years 1990-2020')
    plot_map1(df2,"avg_temp","1960-1990",title = 'Average temperature of stations in years 1960-1990')
    plot_map1(df3,"avg_temp","1930-1960",title = 'Average temperature of stations in years 1930-1960')


    ################################### Question 2-b
    model = PipelineModel.load(model_file)

    ################################### Question 2-b1
    lats, lons = np.meshgrid(np.arange(-90,90,.5),np.arange(-180,180,.5))
    elevs = [eg.get_elevations(np.array([late,lone]).T) for late,lone in zip(lats,lons)]
    elevs_np = np.reshape(elevs, (720,360))
    elev_df = pd.DataFrame({'latitude':lats.reshape(1,-1)[0],'longitude':lons.reshape(1,-1)[0],'elevation':elevs_np.reshape(1,-1)[0]})
    date = datetime.strptime('2019-02-12', '%Y-%m-%d').date()
    elev_df = spark.createDataFrame(elev_df).withColumn("date", functions.lit(date)).withColumn("tmax",functions.lit(1)).cache()

    predictions = model.transform(elev_df)

    plot_map2(predictions,"prediction","pred",date)
    #
    ################################### Question 2-b2
    test_df = spark.read.csv(test_file,schema=schema())
    test_df = model.transform(test_df)
    error_df = test_df.withColumn("error",functions.abs(test_df["prediction"]-test_df["tmax"]))

    plot_map1(error_df,"error","prediction_error",label='prediction Error',title='Error in tmax prediction',cmap='hsv')



if __name__ == '__main__':
    inputs = sys.argv[1]
    model_file = sys.argv[2]
    test_file = sys.argv[3]
    main(inputs,model_file,test_file)
