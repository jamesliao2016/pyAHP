#!/usr/bin/env python3
#coding:utf-8
__author__ = 'lishikun4'
_revise_author__ = 'yanxiangda'

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
# app_name = 'pa_offline_features'
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
    params['update_origin'] = sys.argv[1]
    params['update_end'] = sys.argv[1]
if len(sys.argv) >= 3:
    params['update_end'] = sys.argv[2]
if len(sys.argv) >= 4:
    params['write_mode'] = sys.argv[3]

params['write_mode'] = 'save'
# 销量基线update_end以前的所有历史数据
update_end = params['update_end']
update_start = params['update_origin']

# update_end = '2018-01-20'
# update_start = '2018-04-20'


# 读取各模型以及合并
df_sales_dtcid3 = spark.sql('''
SELECT
    *
FROM
    %s
WHERE
    dt = '%s'
''' % (spa_utils.rename('app.app_pa_sales_dtcid3', params), update_end))

# df_sales_dtcid3 = sql('''select * from app.app_pa_sales_dtcid3  where dt between '2018-01-20' and '2018-04-20' ''')

df_sales_dtcid3 = df_sales_dtcid3.drop('dt')
df_sales_dtcid3.cache()

df_time = spark.sql('''
SELECT
    *
FROM
    %s
WHERE
    dt >= '%s'
AND dt <= '%s'
''' % (spa_utils.rename('app.app_pa_festival_features', params), update_start, update_end))

#df_time = sql('''select * from app.app_pa_festival_features  where dt between '2018-01-20' and '2018-04-20' ''')

df_time = df_time.withColumnRenamed('dt', 'date')
df_time.cache()

# 找出每个cid3有记录的开始时间和结束时间
df_sales_dtcid3_start_end_date = df_sales_dtcid3\
    .groupby('item_third_cate_cd')\
    .agg(F.min('date').alias('start_date'),
         F.max('date').alias('end_date'))

# 填充cid3跨度的日期特征
df_cid3_duration_time = df_sales_dtcid3_start_end_date\
    .join(df_time,
          (df_time['date'] >= df_sales_dtcid3_start_end_date['start_date']) &
          (df_time['date'] <= df_sales_dtcid3_start_end_date['end_date']),
          'left')\
    .drop('start_date', 'end_date')
df_cid3_duration_time.cache()

df_complete =  df_cid3_duration_time\
    .join(df_sales_dtcid3,
          ['date', 'item_third_cate_cd'], 'left')\
    .fillna(0)
df_complete = df_complete\
    .withColumn('dt', F.lit(update_end))\
    .select(['date', 'item_third_cate_cd', 'newyear', 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival', 'nationalday', 'h1111mark', 'h618mark', 'h1212mark', 'week_of_year', 'day_of_year', 'day_of_week', 'sale_qtty', 'after_prefr_amount', 'before_prefr_amount', 'synthetic_before_prefr_amount', 'sku_offer_discount_rate', 'suit_offer_discount_rate', 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate', 'free_gift_discount_rate', 'dq_and_jq_pay_discount_rate', 'jq_pay_discount_rate', 'dq_pay_discount_rate', 'sku_offer_participation_rate', 'suit_offer_participation_rate', 'full_minus_offer_participation_rate', 'ghost_offer_participation_rate', 'free_gift_participation_rate', 'dq_and_jq_pay_participation_rate', 'jq_pay_participation_rate', 'dq_pay_participation_rate', 'dt'])

# df_complete_pan = df_complete.toPandas()
# df_complete_pan.to_csv("pa_features_pan_0529.csv",encoding='utf8',index=False)

spark.sql("set hive.exec.dynamic.partition=true")
spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")
logger.info('inserting app.app_pa_features_dtcid3_0606...')
spa_utils.save_result(df_complete,
                      'app.app_pa_features_cid3_0606',
                      partitioning_columns=['dt'],
                      write_mode=params['write_mode'],
                      spark=spark, params = params)
