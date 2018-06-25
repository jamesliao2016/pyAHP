#!/usr/bin/env python3
# coding:utf-8

import os
import subprocess
import pyspark.sql.functions as F
from pyspark import SparkFiles

def save_result(df, table_name, partitioning_columns=[], write_mode='insert',
                spark=None, params=None):
    if params is None:
        params = dict()
    table_name = rename(table_name, params)
    if isinstance(partitioning_columns, str):
        partitioning_columns = [partitioning_columns]
    save_mode =  'overwrite' if ('overwrite' in params.keys()) and (params['overwrite'] == 1) else 'error'
    if write_mode == 'save':
        if len(partitioning_columns) > 0:
            df.repartition(*partitioning_columns).write.mode(save_mode).partitionBy(partitioning_columns).format('orc').saveAsTable(table_name)
        else:
            df.write.mode(save_mode).format('orc').saveAsTable(table_name)
        spark.sql('''ALTER TABLE %s SET TBLPROPERTIES ('author' = '%s')''' % (table_name, params['author']))
    elif write_mode == 'insert':
        if len(partitioning_columns) > 0:
            rows = df.select(partitioning_columns).distinct().collect()
            querys = []
            for r in rows:
                p_str = ','.join(["%s='%s'" % (k, r[k]) for k in partitioning_columns])
                querys.append("alter table %s drop if exists partition(%s)" %
                              (table_name, p_str))
            for q in querys:
                spark.sql(q)
            df.repartition(*partitioning_columns).write.insertInto(table_name, overwrite=False)
        else:
            df.write.insertInto(table_name, overwrite=False)
    else:
        raise ValueError('mode "%s" not supported ' % write_mode)

def convert_timestamp_to_date(df, cols):
    if type(cols) is not list:
        cols = [cols]
    for col in cols:
        df = df.withColumn(col, F.to_date(col))
    return df


def read_csv_by_dt(table, dt=None, start=None, end=None, spark=None, params=None, **kwargs):
    table_path = os.path.join(params['input_path'], table)
    paths = []
    existed = get_partition(table_path)
    if (dt is None) and (start is None) and (end is None):
        paths.extend(existed)
    elif dt is not None:
        path = os.path.join(table_path, dt)
        paths.append(path)
    elif (start is not None) or (end is not None):
        if start is None:
            start = '0000-00-00'
        if end is None:
            end = 'latest'
        start_path = os.path.join(table_path, start)
        end_path = os.path.join(table_path, end)
        between_path = [x for x in existed if start_path <= x <= end_path ]
        paths.extend(between_path)
    df = spark.read.csv(paths, **kwargs)
    return df


def get_partition(table):
    cmd = "hadoop fs -ls '%s'" % table
    out_bytes = subprocess.check_output(cmd, shell=True)
    out_string = out_bytes.decode('utf-8')
    if len(out_string) == 0:
        return None
    else:
        out_records = out_string.split('\n')
        out_files = [x.split(' ')[-1] for x in out_records[1:-1]]
        return out_files


def read_hive_by_dt(table, dt=None, start=None, end=None, spark=None, params=None):
    if (dt is None) and (start is None) and (end is None):
        start = '0000-00-00'
        end = '9999-99-99'
    elif dt == 'latest':
        start = params['update_end']
        end = '9999-99-99'
    elif dt is not None:
        start = dt
        end = dt
    elif (start is not None) or (end is not None):
        if start is None:
            start = '0000-00-00'
        if end is None:
            end = '9999-99-99'
    tail = params['update_end']
    df = spark.sql(params['querys'][table].format(start=start, end=end, tail=tail))
    return df


def read_table(table, dt=None, start=None, end=None, spark=None, params=None, **kwargs):
    if params['input_path'] == 'hive':
        df = read_hive_by_dt(table, dt=dt, start=start, end=end, spark=spark, params=params)
    elif params['input_path'].startswith('hdfs://'):
        df = read_csv_by_dt(table, dt=dt, start=start, end=end, spark=spark, params=params, **kwargs)
    else:
        raise ValueError('input_path "%s" not supported ' % params['input_path'])
    return df


def load_params():
    try:
        import yaml
        params = yaml.load(open(SparkFiles.get('params.yaml')))
    except ImportError:
        from params import params
    return params


def rename(x, params):
    """
    在配置文件中，self2pop变量必须为每一个表名配置一个pop的别名
    为了避免数据污染，不允许pop表与自营表同名，目前通过判断表明中是否包含pop来识别
    """
    if params and 'self2pop' in params:
        y = params['self2pop'][x]
        if 'pop_' in y:
            return y
        else:
            raise ValueError('table name "%s" not valid ' % y)
    else:
        return x
