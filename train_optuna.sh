model_id=transformer_v1_001
model_name=transformer
trainset=/data1/peizhongyi/applications/Goldwind_2020_Guangxi_CMA_001_trainset_target.csv
validset=/data1/peizhongyi/applications/Goldwind_2020_Guangxi_CMA_001_validset_target.csv
testset=/data1/peizhongyi/applications/Goldwind_2020_Guangxi_CMA_001_testset_target.csv
datetime_col=dtime
feature_cols=stn_id,latitude,longitude,ghi_sfc,rain_sfc,snow_sfc,tskin_sfc,tdew2m_sfc,clflo_sfc,clfmi_sfc,clfhi_sfc,wspd_10,wdir_10,t_10,p_10,rhoair_10,rh_10,wspd_30,wdir_30,t_30,p_30,rhoair_30,rh_30,wspd_50,wdir_50,t_50,p_50,rhoair_50,rh_50,wspd_70,wdir_70,t_70,p_70,rhoair_70,rh_70,wspd
target_cols=wspd
seq_len=48
label_len=12
pred_len=48
interval=900
timestamp_feature=h
gpu=0
epoch=50

cmd="python hpo_train_optuna.py \
    --stage train \
    --model_id ${model_id} \
    --model_name ${model_name} \
    --log_path log \
    --trainset_csv_path ${trainset} \
    --validset_csv_path ${validset} \
    --testset_csv_path ${testset} \
    --datetime_col ${datetime_col} \
    --feature_cols ${feature_cols} \
    --target_cols ${target_cols} \
    --seq_len ${seq_len} \
    --label_len ${label_len} \
    --pred_len ${pred_len} \
    --interval ${interval} \
    --timestamp_feature ${timestamp_feature} \
    --train_epochs ${epoch} \
    --gpu ${gpu} "

log=log/nohup_${model_id}_optuna_set.output
nohup $cmd >> $log &
train_pid=$!
echo -----cmd------
echo ${cmd}
echo ----output----
echo "tail -f $log"

