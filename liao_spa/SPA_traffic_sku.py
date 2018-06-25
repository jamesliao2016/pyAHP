#!/usr/bin/env python3
# coding:utf-8
__author__ = 'lishikun4'

import sys
import os
import yaml
import datetime
import logging
import logging.config
import spa_utils
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark import SparkFiles
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
import pyspark.sql.types as sql_type

app_name = os.path.basename(__file__)
spark = SparkSession.builder.appName(app_name).enableHiveSupport().getOrCreate()
spark.sparkContext.addPyFile('logging.conf')
spark.sparkContext.addPyFile('params.yaml')

logging.config.fileConfig(SparkFiles.get('logging.conf'))  # 读取logging的配置信息
logger = logging.getLogger(app_name)  # logger名称与任务名称相同

# 读取任务参数
params = spa_utils.load_params()
logger.info('parameter file loaded')
logger.debug(params)

# 命令行参数有更高的优先级
if len(sys.argv) >= 2:
    params['update_start'] = sys.argv[1]
    params['update_end'] = sys.argv[1]
if len(sys.argv) >= 3:
    params['update_end'] = sys.argv[2]
if len(sys.argv) >= 4:
    params['write_mode'] = sys.argv[3]

# 读取start date和end date
update_start = params['update_start']
update_end = params['update_end']

traffic_schema = StructType([
    StructField('op_time', sql_type.StringType()),
    StructField('item_sku_id', sql_type.StringType()),
    StructField('sku_name', sql_type.StringType()),
    StructField('web_site_id', sql_type.IntegerType()),
    StructField('pv', sql_type.LongType()),
    StructField('uv', sql_type.LongType()),
    StructField('visits', sql_type.LongType()),
    StructField('dt', sql_type.StringType())
])

# 读取流量表
traffic_df =  spa_utils.read_table('app_cmo_ol_client_sku_3_to_bjmart_di', start = update_start, end = update_end,
                                     spark=spark, params=params, sep='\t', header=True,
                                     schema=traffic_schema)

# app.app_pa_traffic_dtsku
# sku流量模型
# 粒度(dt, sku)
df_sku_traffic = traffic_df\
    .groupBy(['item_sku_id', 'dt'])\
    .agg(
    F.sum('pv').alias('pv'),
    F.sum('uv').alias('uv')
)
df_sku_traffic = df_sku_traffic.select(['item_sku_id', 'pv', 'uv', 'dt'])

spark.sql("set hive.exec.dynamic.partition=true")
spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")
logger.info('inserting app.app_pa_traffic_dtsku...')
spa_utils.save_result(df_sku_traffic,
                      'app.app_pa_traffic_dtsku',
                      partitioning_columns=['dt'],
                      write_mode=params['write_mode'],
                      spark=spark, params = params)