#!/usr/bin/env python3
# coding:utf-8

__revise_author__ = 'yanxiangda'


from pyspark.sql import functions as F
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor

def add_fourier_terms(df, period, col, degree_fourier):
    """根据输入的周期和列的值，添加相应的傅里叶项来描述周期性
    """
    for i in range(1, degree_fourier + 1):
        df = df.withColumn(col + '_fourier_sin_' + str(i),
                           F.sin((2 * np.pi * F.col(col) / period) * i))
        df = df.withColumn(col + '_fourier_cos_' + str(i),
                           F.cos((2 * np.pi * F.col(col) / period) * i))
    return df

def add_datediff(df, date_col, start_date):
    """添加日期序号
    """
    df = df.withColumn('datediff', F.datediff(F.col(date_col), F.lit(start_date)))
    df = df.withColumn('datediff_square', F.pow(F.col('datediff'), 2))
    df = df.withColumn('datediff_square_root', F.pow(F.col('datediff'), 0.5))
    return df

def adjust_promo_features(df, threshold):
    threshold_coupon = 1- threshold
    # 比较成交价与基线价，成交价过高（大于基线价的95%）即认为是假促销
    # 可直接比较优惠后金额与基于基线价格的优惠前金额
    # 假促销的情况下，促销标记变为零（无促销）
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
    # 优惠券的优惠过小的情况下，不考虑优惠券的影响
    df = df\
        .withColumn('dq_and_jq_pay_flag', F.when(
        (F.col('dq_and_jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount')) &
        (F.col('dq_and_jq_pay_flag')), 1).otherwise(0))
    df = df\
        .withColumn('dq_and_jq_pay_discount_rate', F.when(
        F.col('dq_and_jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount'),
        F.col('dq_and_jq_pay_discount_rate')).otherwise(0))
    df = df\
        .withColumn('dq_pay_flag', F.when(
        (F.col('dq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount')) &
        (F.col('dq_pay_flag')), 1).otherwise(0))
    df = df\
        .withColumn('dq_pay_discount_rate', F.when(
        F.col('dq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount'),
        F.col('dq_pay_discount_rate')).otherwise(0))
    df = df\
        .withColumn('jq_pay_flag', F.when(
        (F.col('jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount')) &
        (F.col('jq_pay_flag')), 1).otherwise(0))
    df = df\
        .withColumn('jq_pay_discount_rate', F.when(
        F.col('jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount'),
        F.col('jq_pay_discount_rate')).otherwise(0))
    # 非促销标记做相应调整
    df = df\
        .withColumn('non_promo_flag', F.when(
        ((F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) |
        (F.col('jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount'))) &
        F.col('non_promo_flag'), 1).otherwise(0))
    return df

def convert_boolean_to_int(df, cols):
    if type(cols) is not list:
        cols = [cols]
    for col_to_change in cols:
        df = df.withColumn(col_to_change, F.when(F.col(col_to_change), 1).otherwise(0))
    return df

def adjust_out_of_stock(df):
    """调整矛盾的缺货标记，即有销量但标志为缺货的天
    """
    df = df.withColumn('out_of_stock_flag', F.when(~F.col('out_of_stock_flag') & (F.col('sale_qtty') > 0)))


def calculate_baseline_sku(row, data_source = 'self'):
    model_type = 0
    raw_data = row[1]
    return model_fit(raw_data, data_source, model_type)


def calculate_baseline_cid3(row, data_source = 'self', fi_pd=None):
    model_type = 1
    raw_data = row[1]
    return model_fit(raw_data, data_source, model_type, fi_pd)

'''
def model_fit(raw_data, model_type):
    """数据转换为Pandas DataFrame，并按天排序
    """
    if model_type == 0:
        dataset = pd.DataFrame(list(raw_data), columns = DATASET_SCHEMA_SKU).sort_values('dt').drop_duplicates().reset_index(drop = True)
        result = fit_baseline_sku(dataset)
    elif  model_type == 1:
        dataset = pd.DataFrame(list(raw_data), columns = DATASET_SCHEMA_CID3).sort_values('date').drop_duplicates().reset_index(drop = True)
        result = fit_baseline_cid3(dataset)
    return result.to_dict(orient = 'records')
'''

def model_fit(raw_data, data_source, model_type,fi_pd=None):
    """数据转换为Pandas DataFrame，并按天排序
    """
    if model_type == 0:
        sku_input = input_sku_baseline(raw_data, data_source = data_source)
        result = fit_baseline_sku(sku_input)
    elif  model_type == 1:
        cid_input = input_cid_baseline(raw_data, data_source = data_source)
        result = fit_baseline_cid3(cid_input) if data_source == '7fresh' else revised_fit_baseline_cid3(cid_input, fi_pd)
    return result.to_dict(orient = 'records')

def fit_baseline_sku(sku_input):
    """计算sku的基线销量
    """
    dataset = sku_input.dataset
    try:
        x_df = dataset[sku_input.X_SCHEMA_SKU]
        y_df = dataset[sku_input.Y_SCHEMA_SKU]
        # 剔除outliers。包括缺货的天，无促销但销量很高的天
        cutpoint = np.percentile(dataset['sale_qtty'], 90)
        exclusion_flag = np.where((dataset['out_of_stock_flag'] == 1) |
                                  ((dataset['sale_qtty'] > cutpoint) & (dataset['non_promo_flag'] == 1)),
                                  1, 0)
        x_clean_df = x_df.loc[exclusion_flag == 0]
        y_clean_df = y_df.loc[exclusion_flag == 0]
        x_np = x_clean_df.values
        y_np = y_clean_df.values.ravel()
        # 拟合模型并提取参数
        lm = ElasticNetCV()
        lm.fit(x_np, y_np)
        coefficients = lm.coef_
        intercept = lm.intercept_
        initial_fitting = lm.predict(x_df)
        initial_fitting = np.exp(initial_fitting)
        # 去除促销相关的变量的影响，把相关变量参数设为0
        coefficients_wno_promo = no_promo_coefficients(x_df.columns.values, coefficients, sku_input.sku_promotion_features)
        n_day = x_df.shape[0]
        # 计算初始基线
        initial_base_line = np.dot(x_df, coefficients_wno_promo)
        initial_base_line = initial_base_line + np.ones(n_day) * intercept
        initial_base_line = np.exp(initial_base_line)
        # 平滑初始基线
        smoothed_initial_base_line = sql_ewma(initial_base_line)[0]
        # 计算周期性因子
        seasonality_factor = initial_base_line / initial_base_line.mean()
        # 计算剔除季节性因子的真实销量
        deseasonalized_unit_volume = np.exp(y_df.values.ravel()) / seasonality_factor
        # 促销日、高销量日以及缺货日销量替换为初始基线的均值
        psrb = initial_base_line / seasonality_factor
        day_to_replace = np.where((dataset['sale_qtty'] > cutpoint) |
                             (dataset['out_of_stock_flag'] == 1),
                             0, dataset['non_promo_flag'])
        # 去季节性因素、异常天因素的销量
        deaseasonalized_non_promo_volume = np.where(day_to_replace == 1, deseasonalized_unit_volume, psrb)
        smoothed_deseasonalized_non_promo_unit_volume = sql_ewma(deaseasonalized_non_promo_volume)[0]
        # 最终基线结果
        final_base_line = smoothed_deseasonalized_non_promo_unit_volume * seasonality_factor
        # 保存结果
        output_df = dataset.copy()
        output_df['initial_fitting'] = initial_fitting
        output_df['initial_baseline'] = initial_base_line
        output_df['smoothed_initial_baseline'] = smoothed_initial_base_line
        output_df['deseasonalized_sale_qtty'] = deseasonalized_unit_volume
        output_df['deaseasonalized_nonpromo_sale_qtty'] = deaseasonalized_non_promo_volume
        output_df['smoothed_deseasonalized_nonpromo_sale_qtty'] = smoothed_deseasonalized_non_promo_unit_volume
        output_df['final_baseline'] = np.where(final_base_line >= 0, final_base_line, 0)
        # 计算拉升和拉升率
        output_df['uplift'] = np.where(output_df['non_promo_flag'] == 1, 0,
                                       output_df['sale_qtty'] - output_df['final_baseline'])
        output_df['uplift'] = np.where(output_df['uplift'] > 0, output_df['uplift'], 0)
        output_df['uplift_rate'] = output_df['uplift'] / output_df['sale_qtty']
        # 缺失值定义为0
        output_df['uplift'] = output_df['uplift'].fillna(0)
        output_df['uplift_rate'] = output_df['uplift_rate'].fillna(0)
        output_df = output_df[sku_input.OUTPUT_SCHEMA_SKU]
    except:
        output_df = dataset[sku_input.OUTPUT_KEYS_SKU]
        for col in sku_input.OUTPUT_VALUES_SKU:
            output_df[col] = np.nan
    return output_df

'''
没有删除该函数，但是新的方法不适用该函数
'''
def fit_baseline_cid3(cid_input):
    """计算品类的基线
    """
    dataset = cid_input.dataset
    try:
        x_df = dataset[cid_input.X_SCHEMA_CID3]
        y_df = dataset[cid_input.Y_SCHEMA_CID3]
        x_np = x_df.values
        y_np = y_df.values.ravel()
        reg = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=6)
        fitted_model = reg.fit(x_np, y_np)
        initial_fitting = fitted_model.predict(x_np)
        initial_fitting = np.exp(initial_fitting)
        # 去除促销相关的变量的影响，把相关变量设为0
        x_non_promo = non_promo_inputs(x_df, cid_input.cid3_promotion_features).values
        initial_base_line = fitted_model.predict(x_non_promo)
        initial_base_line = np.exp(initial_base_line)
        final_base_line = sql_ewma(initial_base_line)[0]
        output_df = dataset.copy()
        output_df['initial_fitting'] = initial_fitting
        output_df['initial_baseline'] = initial_base_line
        output_df['final_baseline'] = final_base_line
        output_df = output_df[cid_input.OUTPUT_SCHEMA_CID3]
    except:
        output_df = dataset[cid_input.OUTPUT_KEYS_CID3]
        for col in cid_input.OUTPUT_VALUES_CID3:
            output_df[col] = np.nan
    return output_df


def revised_fit_baseline_cid3(cid_input, fi_pd):
    """修改三级品类计算的算法,
    其中新增两个函数local_randomforest()与local_elasticnet()
    local_randomforest()用于拟合计算节假日相关曲线
    与local_elasticnet()用于拟合计算平常日相关曲线
    """
    dataset = cid_input.dataset
    try:
        #df_list用于将拆分的数据收集，df_list[-1]代表非节假日天的数据集
        df_list = []
        # i = 'newyear'
        for i in cid_input.SPLIT_FESTIVAL:
            current_list = fi_pd[fi_pd['festival_name']==i].reset_index(drop=True)
            #建立空的dataframe，之后进行concat方便
            df_empty = pd.DataFrame(columns=dataset.columns)
            for j in range(len(current_list)):
                current_date = dataset[(dataset['date']>=str(current_list.loc[j,'cal_start_date'])) & (dataset['date']<=str(current_list.loc[j,'cal_end_date']))]
                df_empty = pd.concat([df_empty,current_date])
                dataset = dataset.drop(current_date.index)
            df_list.append(df_empty)
        
        #将normal数据加入数据
        df_list.append(dataset)
        empty_df = pd.DataFrame(columns=cid_input.LOCAL_SCHEMA_CID3)
        #circle_randomforest_al
        for i in df_list[0:-1]:
            empty_df = pd.concat([empty_df,local_randomforest(i, cid_input)])
        
        empty_df=pd.concat([empty_df,local_elasticnet(df_list[-1], cid_input)])
        empty_df = empty_df.sort_values(by="date")
        final_base_line = sql_ewma(empty_df['initial_base_line'])[0]
        empty_df['final_baseline'] = final_base_line
        empty_df = empty_df[cid_input.OUTPUT_SCHEMA_CID3]
    except:
        empty_df = dataset[cid_input.OUTPUT_KEYS_CID3]
        for col in cid_input.OUTPUT_VALUES_CID3:
            empty_df[col] = np.nan
    return empty_df


def local_randomforest(dataset, cid_input):
    x_df = dataset[cid_input.X_SCHEMA_CID3]
    y_df = dataset[cid_input.Y_SCHEMA_CID3]
    x_np = x_df.values
    y_np = y_df.values.ravel()
    reg = RandomForestRegressor(n_estimators=200, criterion='mse', max_depth=10,oob_score=True)
    fitted_model = reg.fit(x_np, y_np)
    initial_fitting = fitted_model.predict(x_np)
    initial_fitting = np.exp(initial_fitting)
    # 去除促销相关的变量的影响，把相关变量设为0
    x_non_promo = non_promo_inputs(x_df, cid_input.cid3_promotion_features).values
    initial_base_line = fitted_model.predict(x_non_promo)
    initial_base_line = np.exp(initial_base_line)
    current_dt = dataset.copy()
    current_dt['initial_base_line'] = initial_base_line
    return current_dt[cid_input.LOCAL_SCHEMA_CID3]


def local_elasticnet(dataset, cid_input):
    x_df = dataset[cid_input.X_SCHEMA_CID3]
    y_df = dataset[cid_input.Y_SCHEMA_CID3]
    #x_df = (x_df - x_df.mean()) / (x_df.max() - x_df.min())
    #elasticnet绝对不能用上面方法进行数据归一化
    x_np = x_df.values
    y_np = y_df.values.ravel()
    lm = ElasticNetCV()
    lm.fit(x_np, y_np)
    coefficients = lm.coef_
    intercept = lm.intercept_
    initial_fitting = lm.predict(x_df)
    initial_fitting = np.exp(initial_fitting)
    x_non_promo = non_promo_inputs(x_df, cid_input.cid3_promotion_features).values
    s_initial_fitting = lm.predict(x_non_promo)
    s_initial_fitting = np.exp(s_initial_fitting)
    current_dt = dataset.copy()
    current_dt['initial_base_line'] = s_initial_fitting
    return current_dt[cid_input.LOCAL_SCHEMA_CID3]

def sql_ewma(vol):
    """ OW平滑函数
    Calculate a non-quite-exponential weighted moving average; the input must be an ordered sequence
    """
    # NB: pd.algos.roll_sum was deprecated, therefore we have to revert to the other version,
    # which is also slated for deprecation
    vol = vol.astype('float64')
    # need to convert NaNs to zeros, otherwise we get NaNs instead of
    # zeros when the sliding window contains only NaNs
    pad = np.zeros(16)
    vol_exp = np.nan_to_num(np.hstack((vol, pad)))
    sum1 = pd.rolling_sum(vol_exp, 34, 0)[8:-8]
    sum2 = pd.rolling_sum(vol_exp, 22, 0)[5:-11]
    sum3 = pd.rolling_sum(vol_exp, 14, 0)[3:-13]
    sum2 = sum2 * 0.75
    sum3 = sum3 * 0.75
    # we use the count of non-NaN weeks as the weights
    # obtain the counts by converting NaN to 0, other to 1, then summing
    pad = np.zeros(16)
    nonpromo_indicator = 1 - np.isnan(vol)
    vol_exp = np.hstack((nonpromo_indicator, pad))
    weight1 = pd.rolling_sum(vol_exp, 34, 0)[8:-8]
    weight2 = pd.rolling_sum(vol_exp, 22, 0)[5:-11]
    weight3 = pd.rolling_sum(vol_exp, 14, 0)[3:-13]
    weight2 = weight2 * 0.75
    weight3 = weight3 * 0.75
    sums = sum1 + sum2 + sum3
    weights = weight1 + weight2 + weight3
    # result is zero where we have zero weeks to sum
    ma = np.zeros(len(weights))
    valid = weights > 0
    ma[valid] = sums[valid] / weights[valid]
    # weight1 also happens to be the numweeks metric that the SQL uses
    return ma, weight1

def no_promo_coefficients(variable_list, coefficients, reset_variable_list):
    """Sets coefficients relating to promotions to zeros
    """
    j = 0
    coefficients_wno_promo_temp = []
    for i in variable_list:
        variable = i.lower()
        if i in reset_variable_list:
            coefficients_wno_promo_temp.append(0)
        else :
            coefficients_wno_promo_temp.append(coefficients[j])
        j += 1
    return coefficients_wno_promo_temp

def non_promo_inputs(df, reset_variable_list):
    """Sets input columns relating to promotions to zeros
    """
    df_temp = df.copy()
    for i in df_temp.columns.values:
        variable = i.lower()
        if i in reset_variable_list:
            df_temp[i] = 0.0
    return df_temp


class input_sku_baseline:
    def __init__(self, raw_data, data_source='self'):
        self.data_source = data_source
        if data_source == '7fresh':
            # 7fresh sku schema
            self.DATASET_SCHEMA_SKU = ['store_id', 'sku_id', 'newyear', 'springfestival', 'tombsweepingfestival', 'labourday',
                    'dragonboatfestival', 'midautumnfestival', 'nationalday', 'week_of_year', 'day_of_year',
                    'day_of_week', 'sku_offer_flag', 'full_minus_offer_flag', 'discount_code_offer_flag',
                    'free_goods_flag', 'ghost_offer_flag', 'coupon_pay_flag', 'sale_qtty', 'after_offer_amount',
                    'before_prefr_amount', 'synthetic_before_prefr_amount', 'coupon_pay_amount', 'major_offer_amount',
                    'sku_offer_amount', 'full_minus_offer_amount', 'discount_code_offer_amount',
                    'synthetic_total_offer_amount', 'free_goods_amount', 'sale_qtty_for_full_minus',
                    'sale_qtty_for_coupon_pay', 'before_prefr_amount_for_free_gift',
                    'before_prefr_amount_for_major_offer', 'before_prefr_amount_for_coupon_pay', 'count_sale_ord_id',
                    'non_promo_flag', 'participation_rate_full_minus', 'participation_rate_coupon_pay',
                    'sku_offer_discount_rate', 'full_minus_offer_discount_rate', 'discount_code_offer_discount_rate',
                    'ghost_offer_discount_rate', 'coupon_pay_discount_rate', 'free_gift_discount_rate',
                    'out_of_stock_flag', 'dt', 'log_sale_qtty', 'datediff', 'datediff_square', 'datediff_square_root',
                    'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1', 'week_of_year_fourier_sin_2',
                    'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3',
                    'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
                    'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.X_SCHEMA_SKU = ['sku_offer_flag', 'full_minus_offer_flag', 'free_goods_flag', 'ghost_offer_flag',
                                 'discount_code_offer_flag', 'coupon_pay_flag', 'participation_rate_full_minus',
                                 'participation_rate_coupon_pay',
                                 'sku_offer_discount_rate', 'discount_code_offer_discount_rate',
                                 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                 'coupon_pay_discount_rate',
                                 'free_gift_discount_rate', 'newyear',
                                 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival',
                                 'midautumnfestival',
                                 'nationalday', 'day_of_week', 'out_of_stock_flag',
                                 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1',
                                 'week_of_year_fourier_sin_2',
                                 'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3',
                                 'week_of_year_fourier_cos_3',
                                 'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
                                 'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.Y_SCHEMA_SKU = ['log_sale_qtty']
            self.OUTPUT_KEYS_SKU = ['dt', 'sku_id', 'store_id']
            self.OUTPUT_VALUES_SKU = ['final_baseline', 'uplift', 'uplift_rate']
            self.OUTPUT_SCHEMA_SKU = ['dt', 'sku_id', 'store_id', 'final_baseline', 'uplift', 'uplift_rate']
            self.sku_promotion_features = ['sku_offer_flag', 'full_minus_offer_flag', 'free_goods_flag',
                                           'ghost_offer_flag', 'discount_code_offer_flag',
                                           'coupon_pay_flag', 'participation_rate_full_minus',
                                           'participation_rate_coupon_pay', 'sku_offer_discount_rate',
                                           'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                           'coupon_pay_discount_rate', 'free_gift_discount_rate',
                                           'discount_code_offer_discount_rate']
        else:
            # self sku schema
            self.DATASET_SCHEMA_SKU = ['item_sku_id', 'newyear', 'springfestival', 'tombsweepingfestival',
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
            self.X_SCHEMA_SKU = ['sku_offer_flag', 'suit_offer_flag', 'full_minus_offer_flag', 'free_gift_flag',
                                 'ghost_offer_flag', 'dq_and_jq_pay_flag', 'non_promo_flag',
                                 'participation_rate_full_minus_and_suit_offer', 'sku_offer_discount_rate',
                                 'full_minus_offer_discount_rate', 'suit_offer_discount_rate',
                                 'ghost_offer_discount_rate',
                                 'dq_and_jq_pay_discount_rate', 'free_gift_discount_rate', 'newyear', 'springfestival',
                                 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival',
                                 'nationalday',
                                 'h1111mark', 'h618mark', 'h1212mark', 'day_of_week', 'out_of_stock_flag',
                                 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1',
                                 'week_of_year_fourier_sin_2',
                                 'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3',
                                 'week_of_year_fourier_cos_3',
                                 'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
                                 'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.Y_SCHEMA_SKU = ['log_sale_qtty']
            self.OUTPUT_KEYS_SKU = ['dt', 'item_sku_id']
            self.OUTPUT_VALUES_SKU = ['final_baseline', 'uplift', 'uplift_rate']
            self.OUTPUT_SCHEMA_SKU = ['dt', 'item_sku_id', 'final_baseline', 'uplift', 'uplift_rate']
            self.sku_promotion_features = ['sku_offer_flag', 'suit_offer_flag', 'full_minus_offer_flag',
                                           'free_gift_flag',
                                           'ghost_offer_flag', 'dq_and_jq_pay_flag',
                                           'participation_rate_full_minus_and_suit_offer',
                                           'participation_rate_dq_and_jq_pay',
                                           'sku_offer_discount_rate', 'full_minus_offer_discount_rate',
                                           'suit_offer_discount_rate', 'ghost_offer_discount_rate',
                                           'dq_and_jq_pay_discount_rate', 'free_gift_discount_rate']
        self.dataset = pd.DataFrame(list(raw_data), columns=self.DATASET_SCHEMA_SKU).sort_values('dt').drop_duplicates().reset_index(drop=True)

class input_cid_baseline:
    def __init__(self, raw_data, data_source='self'):
        self.data_source = data_source
        if data_source == '7fresh':
            # 7fresh cid4 schema
            self.DATASET_SCHEMA_CID3 = ['date', 'store_id', 'cate_id_4', 'newyear', 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival', 'nationalday', 'week_of_year', 'day_of_year', 'day_of_week', 'sale_qtty', 'after_offer_amount', 'before_prefr_amount', 'synthetic_before_prefr_amount', 'synthetic_sku_offer_amount', 'synthetic_discount_code_offer_amount', 'synthetic_full_minus_offer_amount', 'synthetic_ghost_offer_amount', 'free_gift_offer_amount', 'coupon_pay_amount', 'sku_offer_sale_qtty', 'discount_code_offer_sale_qtty', 'full_minus_offer_sale_qtty', 'ghost_offer_sale_qtty', 'free_gift_sale_qtty', 'coupon_pay_sale_qtty', 'sku_offer_discount_rate', 'discount_code_offer_discount_rate', 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate', 'free_gift_discount_rate', 'coupon_pay_discount_rate', 'sku_offer_participation_rate', 'discount_code_offer_participation_rate', 'full_minus_offer_participation_rate', 'ghost_offer_participation_rate', 'free_gift_participation_rate', 'coupon_pay_participation_rate', 'log_synthetic_before_prefr_amount', 'datediff', 'datediff_square', 'datediff_square_root', 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1', 'week_of_year_fourier_sin_2', 'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3', 'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2', 'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.X_SCHEMA_CID3 = ['newyear', 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival', 'nationalday', 'day_of_week', 'sku_offer_discount_rate', 'discount_code_offer_discount_rate', 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate', 'free_gift_discount_rate', 'coupon_pay_discount_rate', 'sku_offer_participation_rate', 'discount_code_offer_participation_rate', 'full_minus_offer_participation_rate', 'ghost_offer_participation_rate', 'free_gift_participation_rate', 'coupon_pay_participation_rate', 'datediff', 'datediff_square', 'datediff_square_root', 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1', 'week_of_year_fourier_sin_2', 'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3', 'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2', 'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.Y_SCHEMA_CID3 = ['log_synthetic_before_prefr_amount']
            self.OUTPUT_KEYS_CID3 = ['date', 'store_id', 'cate_id_4']
            self.OUTPUT_VALUES_CID3 = ['final_baseline']
            self.OUTPUT_SCHEMA_CID3 = ['date', 'store_id', 'cate_id_4', 'final_baseline']
            self.cid3_promotion_features = ['sku_offer_discount_rate', 'discount_code_offer_discount_rate', 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate', 'free_gift_discount_rate', 'coupon_pay_discount_rate', 'sku_offer_participation_rate', 'discount_code_offer_participation_rate', 'full_minus_offer_participation_rate', 'ghost_offer_participation_rate', 'free_gift_participation_rate', 'coupon_pay_participation_rate']
            self.LOCAL_SCHEMA_CID3 = ['date', 'store_id', 'cate_id_4', 'initial_base_line']
            self.SPLIT_FESTIVAL = []
        else:
            # self cid3 schema
            self.DATASET_SCHEMA_CID3 = ['date', 'item_third_cate_cd', 'newyear', 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival', 'nationalday', 'h1111mark', 'h618mark', 'h1212mark', 'week_of_year', 'day_of_year', 'day_of_week', 'sale_qtty', 'after_prefr_amount', 'before_prefr_amount', 'synthetic_before_prefr_amount', 'sku_offer_discount_rate', 'suit_offer_discount_rate', 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate', 'free_gift_discount_rate', 'dq_and_jq_pay_discount_rate', 'jq_pay_discount_rate', 'dq_pay_discount_rate', 'sku_offer_participation_rate', 'suit_offer_participation_rate', 'full_minus_offer_participation_rate', 'ghost_offer_participation_rate', 'free_gift_participation_rate', 'dq_and_jq_pay_participation_rate', 'jq_pay_participation_rate', 'dq_pay_participation_rate', 'log_synthetic_before_prefr_amount', 'datediff', 'datediff_square', 'datediff_square_root', 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1', 'week_of_year_fourier_sin_2', 'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3', 'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2', 'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.X_SCHEMA_CID3 = ['sku_offer_discount_rate', 'suit_offer_discount_rate', 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate', 'free_gift_discount_rate', 'dq_and_jq_pay_discount_rate', 'sku_offer_participation_rate', 'suit_offer_participation_rate', 'full_minus_offer_participation_rate', 'ghost_offer_participation_rate', 'free_gift_participation_rate', 'dq_and_jq_pay_participation_rate', 'newyear', 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival', 'nationalday', 'h1111mark', 'h618mark', 'h1212mark', 'week_of_year', 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1', 'week_of_year_fourier_sin_2', 'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3', 'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2', 'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.Y_SCHEMA_CID3 = ['log_synthetic_before_prefr_amount']
            self.OUTPUT_KEYS_CID3 = ['date', 'item_third_cate_cd']
            self.OUTPUT_VALUES_CID3 = ['final_baseline']
            self.OUTPUT_SCHEMA_CID3 = ['date', 'item_third_cate_cd', 'final_baseline']
            self.cid3_promotion_features = ['sku_offer_discount_rate', 'suit_offer_discount_rate', 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate', 'free_gift_discount_rate', 'dq_and_jq_pay_discount_rate', 'sku_offer_participation_rate', 'suit_offer_participation_rate', 'full_minus_offer_participation_rate', 'ghost_offer_participation_rate', 'free_gift_participation_rate', 'dq_and_jq_pay_participation_rate']
            '''
            新增常量LOCAL_SCHEMA_CID3，用于将local_randomforest与local_elasticnet方法的结果统一
            '''
            self.LOCAL_SCHEMA_CID3 = ['date', 'item_third_cate_cd', 'initial_base_line']
            self.SPLIT_FESTIVAL = ['h1111mark','h618mark','h1212mark']
        self.dataset = pd.DataFrame(list(raw_data), columns=self.DATASET_SCHEMA_CID3).sort_values(
            'date').drop_duplicates().reset_index(drop=True)
