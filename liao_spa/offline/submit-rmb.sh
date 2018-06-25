
pyspark --master yarn --deploy-mode client         --driver-memory 16G --executor-cores 4         --num-executors 200 --executor-memory 16G         --conf spark.default.parallelism=2000 --conf spark.sql.shuffle.partitions=2000

pyspark \
--conf spark.executor.memory=16g \
--conf spark.executor.instances=60 \
--conf spark.default.parallelism=1200 \
--conf spark.sql.shuffle.partitions=1200 \
--conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
--conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
--conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_algorithm:latest \
--conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_algorithm:latest

#!/bin/sh
spark-submit \
--master yarn-cluster \
--conf spark.executor.instances=200 \
--conf spark.executor.memory=32g \
--conf spark.executor.cores=8 \
--conf spark.driver.cores=8 \
--conf spark.driver.memory=16g \
--conf spark.yarn.am.cores=8 \
--conf spark.yarn.am.memory=16g \
--conf spark.default.parallelism=1000 \
--conf spark.sql.shuffle.partitions=1000 \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.network.timeout=600s \
--conf spark.rpc.askTimeout=1200s \
--conf spark.rpc.lookupTimeout=600s \
--conf spark.rpc.numRetries=10 \
--conf spark.shuffle.io.maxRetries=10 \
--conf spark.pyspark.python=python2.7 \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.network.timeout=500 \
--conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
--conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
--conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_algorithm:latest \
--conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_algorithm:latest \
--files params.yaml,params.py,logging.conf,spa_utils.py,SPA_baseline_functions.py \
 SPA_baseline_sku_0607.py


