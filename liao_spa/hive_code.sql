-- #### Create table for the offline base sales data, 29 May 2018

CREATE TABLE `app.app_pa_baseline_cid3_0606`(
  `date` string,
  `item_third_cate_cd` string,
  `final_baseline` double)
COMMENT 'gmv of cid3' PARTITIONED BY(dt STRING) ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE TBLPROPERTIES ('author'='liaopeng10');
