#!/usr/bin/env python3
# coding:utf-8
__author__ = 'lishikun4'

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

# app_name = os.path.basename(__file__)
app_name = 'sku_baseline'
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

selected_columns = ['item_sku_id', 'newyear', 'springfestival', 'tombsweepingfestival',
       'labourday', 'dragonboatfestival', 'midautumnfestival',
       'nationalday', 'h1111mark', 'h618mark', 'h1212mark', 'week_of_year',
       'day_of_year', 'day_of_week', 'free_gift_flag', 'ghost_offer_flag',
       'dq_and_jq_pay_flag', 'jq_pay_flag', 'dq_pay_flag',
       'full_minus_offer_flag', 'suit_offer_flag', 'sku_offer_flag',
       'non_promo_flag', 'sale_qtty', 'after_prefr_amount',
       'before_prefr_amount', 'synthetic_before_prefr_amount',
       'participation_rate_full_minus_and_suit_offer',
       'participation_rate_dq_and_jq_pay', 'sku_offer_discount_rate',
       'full_minus_offer_discount_rate', 'suit_offer_discount_rate',
       'ghost_offer_discount_rate', 'dq_and_jq_pay_discount_rate',
       'jq_pay_discount_rate', 'dq_pay_discount_rate',
       'free_gift_discount_rate', 'out_of_stock_flag', 'dt',
       'log_sale_qtty', 'gross_price', 'datediff', 'datediff_square',
       'datediff_square_root', 'week_of_year_fourier_sin_1',
       'week_of_year_fourier_cos_1', 'week_of_year_fourier_sin_2',
       'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3',
       'week_of_year_fourier_cos_3', 'day_of_week_fourier_sin_1',
       'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
       'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3',
       'day_of_week_fourier_cos_3']
def format_result_sku(row):
    return (
        str(row['dt']),
        int(row['item_sku_id']),
        float(row['final_baseline']),
        float(row['uplift']),
        float(row['uplift_rate'])
    )

SCHEMA_OUTPUT_SKU = StructType([
    StructField("dt", sql_type.StringType()),
    StructField("item_sku_id", sql_type.LongType()),
    StructField("final_baseline", sql_type.DoubleType()),
    StructField("uplift", sql_type.DoubleType()),
    StructField("uplift_rate", sql_type.DoubleType())
])

# 读取订单模型B1
df_features = spark.sql('''
SELECT
    *
FROM %s
WHERE
    dt <= '%s'
''' % (spa_utils.rename('app.app_pa_features_dtsku', params), update_end))
df_features.cache()

# 读取最新的库存状态
df_hierarchical_sku = spark.sql('''
SELECT
    item_sku_id, start_date, sku_status_cd
FROM %s
WHERE
    dt = '%s'
''' % (spa_utils.rename('app.app_pa_hierarchical_sku', params), update_end))
df_hierarchical_sku = df_hierarchical_sku.filter(F.col('sku_status_cd') == '3000').drop('sku_status_cd')
df_hierarchical_sku.cache()

# 剔除每个sku，最近一次状态为下柜时的，下柜时间的数据
df = df_features.join(df_hierarchical_sku, ['item_sku_id'], 'left') \
    .filter((F.col('dt') < F.col('start_date')) | (F.col('start_date').isNull()))\
    .drop('start_date')


# 用销量的对数作为因变量
df = df.withColumn('log_sale_qtty', F.log(F.col('sale_qtty') + 0.0001))

# 计算平均优惠前价格
df = df.withColumn('gross_price', F.when(F.col('sale_qtty') > 0,
                                         F.col('synthetic_before_prefr_amount')/F.col('sale_qtty')).otherwise(0))

# 添加时间趋势项，日期序号、日期序号平方、日期序号平方根
df = add_datediff(df, 'dt', update_start)

# 添加傅里叶参数项
df = add_fourier_terms(df, 53, 'week_of_year', 3)
df = add_fourier_terms(df, 7, 'day_of_week', 3)
#df = adjust_promo_features(df, 0.95)

threshold = 0.95
df = df\
        .withColumn('sku_offer_flag', F.when(
        (F.col('sku_offer_flag').isNotNull()) &
        (F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) &
        (F.col('sku_offer_flag')), 1).otherwise(0))
df = df\
        .withColumn('full_minus_offer_flag', F.when(
        (F.col('full_minus_offer_flag').isNotNull()) &
        (F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) &
        (F.col('full_minus_offer_flag')), 1).otherwise(0))
df = df\
        .withColumn('suit_offer_flag', F.when(
        (F.col('suit_offer_flag').isNotNull()) &
        (F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) &
        (F.col('suit_offer_flag')), 1).otherwise(0))
df = df\
        .withColumn('ghost_offer_flag', F.when(
        (F.col('ghost_offer_flag').isNotNull()) &
        (F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) &
        (F.col('ghost_offer_flag')), 1).otherwise(0))
# 假促销的情况下，促销折扣变为零
df = df\
        .withColumn('sku_offer_discount_rate', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('sku_offer_discount_rate')).otherwise(0))
df = df\
        .withColumn('full_minus_offer_discount_rate', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('full_minus_offer_discount_rate')).otherwise(0))
df = df\
        .withColumn('suit_offer_discount_rate', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('suit_offer_discount_rate')).otherwise(0))
df = df\
        .withColumn('ghost_offer_discount_rate', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('ghost_offer_discount_rate')).otherwise(0))
# 假促销的情况下，满减与套装促销参与度变为零
df = df\
        .withColumn('participation_rate_full_minus_and_suit_offer', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('participation_rate_full_minus_and_suit_offer')).otherwise(0))

df = df\
        .withColumn('dq_and_jq_pay_flag', F.when(
        (F.col('dq_and_jq_pay_flag').isNotNull()) &
        (F.col('dq_and_jq_pay_flag')), 1).otherwise(0))
df = df\
        .withColumn('free_gift_flag', F.when(
        (F.col('free_gift_flag').isNotNull()) &
        (F.col('free_gift_flag')), 1).otherwise(0))

df = df\
    .withColumn('free_gift_discount_rate', F.when(
    F.col('free_gift_discount_rate') < 1, F.col('free_gift_discount_rate')).otherwise(1))


df = df\
        .withColumn('non_promo_flag', F.when(
    (F.col('after_prefr_amount') <= threshold * F.col('synthetic_before_prefr_amount')) &
    ((F.col('sku_offer_flag') + F.col('suit_offer_flag') + F.col('full_minus_offer_flag')) == 0), 1).otherwise(0))

# boolean转换为int作为回归特征
#df = convert_boolean_to_int(df, ['newyear', 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival', 'nationalday', 'h1111mark', 'h618mark', 'h1212mark'])

df.cache()

# 过滤掉数据过少的sku
df_sku_count = df.groupBy('item_sku_id').count().filter(F.col('count') >= 30)
df = df.join(df_sku_count, ['item_sku_id'], 'inner')

df = df.select(selected_columns)
# Dataframe转换为rdd，并按Sku进行分组，分别计算每个sku的销量基线
result = df.rdd.map(lambda row: ((row['item_sku_id']), row)).groupByKey()\
    .flatMap(lambda row : calculate_baseline_sku(row, 'self'))

# 结果保存为Spark Dataframe
result_df = spark.createDataFrame(result.map(format_result_sku), schema = SCHEMA_OUTPUT_SKU)
result_df = result_df.na.drop()
result_df = result_df.withColumnRenamed('dt', 'date').withColumn('dt', F.lit(update_end))\
    .select('date', 'item_sku_id', 'final_baseline', 'uplift', 'uplift_rate', 'dt')

spark.sql("set hive.exec.dynamic.partition=true")
spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")
logger.info('Saving results...')
logger.info('inserting app.app_pa_baseline_sku...')

# def save_result(df, table_name, partitioning_columns=[], repartitioning_columns=[], write_mode='insert',
#                 spark=None, params=None):
#     if params is None:
#         params = dict()
#     table_name = spa_utils.rename(table_name, params)
#     if isinstance(partitioning_columns, str):
#         partitioning_columns = [partitioning_columns]
#     save_mode =  'overwrite' if ('overwrite' in params.keys()) and (params['overwrite'] == 1) else 'error'
#     if write_mode == 'save':
#         if len(partitioning_columns) > 0:
#             df.repartition(*repartitioning_columns).write.mode(save_mode).partitionBy(partitioning_columns).format('orc').saveAsTable(table_name)
#         else:
#             df.write.mode(save_mode).format('orc').saveAsTable(table_name)
#     elif write_mode == 'insert':
#         if len(partitioning_columns) > 0:
#             rows = df.select(partitioning_columns).distinct().collect()
#             querys = []
#             for r in rows:
#                 p_str = ','.join(["%s='%s'" % (k, r[k]) for k in partitioning_columns])
#                 querys.append("alter table %s drop if exists partition(%s)" %
#                               (table_name, p_str))
#             for q in querys:
#                 spark.sql(q)
#             df.repartition(*repartitioning_columns).write.insertInto(table_name, overwrite=False)
#         else:
#             df.write.insertInto(table_name, overwrite=False)
#     else:
#         raise ValueError('mode "%s" not supported ' % write_mode)

# save_result(result_df,
#                       'app.app_pa_baseline_sku_0607',
#                       partitioning_columns=['dt'], repartitioning_columns=['date'],
#                       write_mode=params['write_mode'],
#                       spark=spark, params = params)
result_df.write.insertInto('app.app_p101_baseline_sku_0607',overwrite=True)

logger.info('insert table done')