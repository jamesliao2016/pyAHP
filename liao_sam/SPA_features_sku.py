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

# 读取各模型以及合并
# 销量
df_sales_dtsku = spark.sql('''
SELECT
    *
FROM
    %s
WHERE
    dt >= '%s'
AND dt <= '%s'
''' % (spa_utils.rename('app.app_pa_sales_dtsku', params), update_start, update_end))
df_sales_dtsku.cache()

# 库存
df_stock = spark.sql('''
SELECT
    sku_id, dt, out_of_stock_flag
FROM
    %s
WHERE
    dt >= '%s'
AND dt <= '%s'
''' % (spa_utils.rename('app.app_pa_stock_dtsku', params), update_start, update_end))
df_stock = df_stock\
    .withColumnRenamed('sku_id', 'item_sku_id')
df_stock.cache()

# 时间特征
df_time = spark.sql('''
SELECT
    *
FROM
    %s
WHERE
    dt >= '%s'
AND dt <= '%s'
''' % (spa_utils.rename('app.app_pa_time', params), update_start, update_end))
df_time.cache()

# 由于基线分析所需的时间范围跟sku上下柜的时间可能不一致
# 找出每个sku有记录的开始时间； 之后没有记录的天用0填充
# 在后面基线计算时，会根据最新的上下柜信息筛选数据
df_sales_dtsku_start_end_date = spark.sql('''
SELECT
    item_sku_id, start_date
FROM
    %s
''' % (spa_utils.rename('app.app_pa_sales_duration', params)))
df_sales_dtsku_start_end_date = df_sales_dtsku_start_end_date \
    .withColumn('start_date', F.when(F.col('start_date') >= update_start, F.col('start_date')).otherwise(update_start))
df_sales_dtsku_start_end_date.cache()

# 填充各sku跨度的日期特征
df_sku_duration_time = df_sales_dtsku_start_end_date\
    .join(df_time,
          df_time['dt'] >= df_sales_dtsku_start_end_date['start_date'],
          'left')\
    .drop('start_date')\
    .filter(F.col('dt').isNotNull())
df_sku_duration_time.cache()

# 合并
df_complete =  df_sku_duration_time\
    .join(df_sales_dtsku,
          ['dt', 'item_sku_id'], 'left')\
    .join(df_stock,
          ['dt', 'item_sku_id'], 'left')\
    .fillna(0)
df_complete = df_complete.select(['item_sku_id', 'newyear', 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival', 'nationalday', 'h1111mark', 'h618mark', 'h1212mark', 'week_of_year', 'day_of_year', 'day_of_week', 'free_gift_flag', 'ghost_offer_flag', 'dq_and_jq_pay_flag', 'jq_pay_flag', 'dq_pay_flag', 'full_minus_offer_flag', 'suit_offer_flag', 'sku_offer_flag', 'non_promo_flag', 'sale_qtty', 'after_prefr_amount', 'before_prefr_amount', 'synthetic_before_prefr_amount', 'participation_rate_full_minus_and_suit_offer', 'participation_rate_dq_and_jq_pay', 'sku_offer_discount_rate', 'full_minus_offer_discount_rate', 'suit_offer_discount_rate', 'ghost_offer_discount_rate', 'dq_and_jq_pay_discount_rate', 'jq_pay_discount_rate', 'dq_pay_discount_rate', 'free_gift_discount_rate', 'out_of_stock_flag', 'dt'])

spark.sql("set hive.exec.dynamic.partition=true")
spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")
logger.info('inserting app.app_pa_features_dtsku...')
spa_utils.save_result(df_complete,
                      'app.app_pa_features_dtsku',
                      partitioning_columns=['dt'],
                      write_mode=params['write_mode'],
                      spark=spark, params = params)