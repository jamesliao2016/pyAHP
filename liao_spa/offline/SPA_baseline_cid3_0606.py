#!/usr/bin/env python3
# coding:utf-8


import sys
import os
import yaml
import datetime
import logging
import logging.config
from SPA_baseline_functions import *
import spa_utils
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark import SparkFiles
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
import pyspark.sql.types as sql_type

app_name = os.path.basename(__file__)
# app_name = 'pa_offline'
spark = SparkSession.builder.appName(app_name).enableHiveSupport().getOrCreate()
spark.sparkContext.addPyFile('logging.conf')
spark.sparkContext.addPyFile('params.yaml')
spark.sparkContext.addPyFile('SPA_baseline_functions.py')

logging.config.fileConfig(SparkFiles.get('logging.conf'))  # 读取logging的配置信息
logger = logging.getLogger(app_name)  # logger名称与任务名称相同

# 读取任务参数
params = spa_utils.load_params()
logger.info('parameter file loaded')
logger.debug(params)

# 命令行参数有更高的优先级
if len(sys.argv) >= 2:
    params['update_origin'] = sys.argv[1]
    params['update_end'] = sys.argv[1]
if len(sys.argv) >= 3:
    params['update_end'] = sys.argv[2]
if len(sys.argv) >= 4:
    params['write_mode'] = sys.argv[3]

# 销量基线update_end以前的所有历史数据
update_end = params['update_end']
update_start = params['update_origin']

# update_end = '2018-01-20'
# update_start = '2018-01-20'

def format_result_cid3(row):
    return (
        str(row['date']),
        str(row['item_third_cate_cd']),
        float(row['final_baseline'])
    )

SCHEMA_OUTPUT_CID3 = StructType([
    StructField("date", sql_type.StringType()),
    StructField("item_third_cate_cd", sql_type.StringType()),
    StructField("final_baseline", sql_type.DoubleType())
])


# 读取订单模型B1
df = spark.sql('''
SELECT
    *
FROM %s
WHERE
    dt = '%s'
''' % (spa_utils.rename('app.app_pa_features_dtcid3', params), update_end))
df = df.drop('dt')
df.cache()

# df=spark.read.csv("hdfs:///user/mart_rmb/liaopeng/app_pa_features_dtcid3_0529",encoding='UTF-8')

df = df.drop('dt')
df.cache()

df = df.withColumn('log_synthetic_before_prefr_amount', F.log(F.col('synthetic_before_prefr_amount') + 0.0001))
df = add_datediff(df, 'date', update_start)
df = add_fourier_terms(df, 53, 'week_of_year', 3)
df = add_fourier_terms(df, 7, 'day_of_week', 3)

#df = convert_boolean_to_int(df, ['newyear', 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival', 'nationalday', 'h1111mark', 'h618mark', 'h1212mark'])

# Dataframe转换为rdd，并按item_third_cate_cd进行分组，分别计算每个item_third_cate_cd的销量基线


#result = df.rdd.map(lambda row: ((row['item_third_cate_cd']), row)).groupByKey().flatMap(calculate_baseline_cid3)
fi_df = spark.sql('''
SELECT
    *
FROM %s
''' % (spa_utils.rename('app.app_pa_festival_information', params)))


fi_df = fi_df.toPandas()
result = df.rdd \
    .map(lambda row: ((row['item_third_cate_cd']), row)).groupByKey() \
    .flatMap(lambda row: calculate_baseline_cid3(row, 'self', fi_df))

# 结果保存为Spark Dataframe
result_df = spark.createDataFrame(result.map(format_result_cid3), schema = SCHEMA_OUTPUT_CID3)
result_df = result_df.na.drop()
result_df = result_df\
    .withColumn('dt', F.lit(update_end))\
    .select('date', 'item_third_cate_cd', 'final_baseline','dt')

# result_df_pan = result_df.toPandas()
# result_df_pan.to_csv("result_df_pan_0529.csv",encoding='utf8',index=False)
# result_df_pan.to_csv("hdfs:///user/mart_rmb/liaopeng/result_df_pan_0529.csv",encoding='utf8',index=False)

spark.sql("set hive.exec.dynamic.partition=true")
spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")
logger.info('Saving results...')
logger.info('inserting app.app_pa_baseline_cid3...')
spa_utils.save_result(result_df,
                      'app.app_pa_baseline_cid3_0606',
                      partitioning_columns=['dt'],
                      write_mode=params['write_mode'],
                      spark=spark, params = params)
logger.info('insert table done')