export CUDA_VISIBLE_DEVICES=0
# for target in {'1效蒸发器汽温','1效蒸发器汽室压力','1效原液换热前温度','1效出料温度'}
# do
#     cmd='''python main_nni.py 
#     --stage train_only 
#     --model_id alumina_timexer
#     --model_name alumina_timexer
#     --features M
#     --log_path log 
#     --trainset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_trainset.csv 
#     --validset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_validset.csv
#     --testset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_testset_new.csv 
#     --datetime_col dtime 
#     --feature_cols 5效5闪到调配槽后循环母液Nk,5效5闪到调配槽后循环母液ak/Rp,6效出料密度,5闪出料密度,57效循环母液进料温度,57效循环母液进料密度,冷凝器出汽压力,5效5闪蒸发母液去循环母液调配槽总流量,2效蒸发器汽温,2效蒸发器汽室压力,2效原液换热前温度,3效蒸发器汽温,3效蒸发器汽室压力,3效原液换热前温度,4效蒸发器汽温,4效蒸发器汽室压力,4效原液换热前温度,5效蒸发器汽温,5效蒸发器汽室压力,5效原液换热前温度,5效循环母液进料流量,6效蒸发器汽温,6效蒸发器汽室压力,6效原液换热前温度,原液闪蒸器循环母液进料流量,原液闪蒸器出料温度,原液闪蒸器蒸汽压力,7效原液换热前温度,7效蒸发器汽温,7效蒸发器汽室压力,1闪出料温度,2闪出料温度,3闪出料温度,4闪出料温度,5闪出料温度,5闪去循环母液调配槽密度,5闪去循环母液调配槽流量,新蒸汽进料压力,新蒸汽进料流量,新蒸汽进料温度,循环上水温度,循环上水流量,循环下水温度,原液闪蒸器循环母液进料阀开度,1效出料阀门开度,1闪出料阀开度,3闪出料阀开度,4闪出料阀开度,2效到1效泵开度,4效到3效泵开度,5效到4效泵开度,6效出料出料泵开度,target
#     --target_cols target  
#     --seq_len 30 
#     --label_len 2 
#     --pred_len 30 
#     --timestamp_feature none  
#     --train_epochs 50 
#     --gpu 0 
#     --use_nni
#     --checkpoints ./checkpoints/alumina_transformer_train_only
#     --hidden_dim 128
#     --d_model 128
#     --e_layers 1
#     --d_layers 1
#     '''

#     echo $target
#     $cmd
# done


# 1效蒸发器汽温,1效蒸发器汽室压力,1效原液换热前温度,1效出料温度

cmd1='''python -u main_nni.py 
    --stage train_only 
    --model_id alumina_timexer
    --model_name alumina_timexer
    --features MS
    --log_path log 
    --trainset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_trainset.csv 
    --validset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_validset.csv
    --testset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_testset_new.csv 
    --datetime_col dtime 
    --feature_cols '5效5闪到调配槽后循环母液Nk,5效5闪到调配槽后循环母液ak/Rp,6效出料密度,5闪出料密度,57效循环母液进料温度,57效循环母液进料密度,冷凝器出汽压力,5效5闪蒸发母液去循环母液调配槽总流量,2效蒸发器汽温,2效蒸发器汽室压力,2效原液换热前温度,3效蒸发器汽温,3效蒸发器汽室压力,3效原液换热前温度,4效蒸发器汽温,4效蒸发器汽室压力,4效原液换热前温度,5效蒸发器汽温,5效蒸发器汽室压力,5效原液换热前温度,5效循环母液进料流量,6效蒸发器汽温,6效蒸发器汽室压力,6效原液换热前温度,原液闪蒸器循环母液进料流量,原液闪蒸器出料温度,原液闪蒸器蒸汽压力,7效原液换热前温度,7效蒸发器汽温,7效蒸发器汽室压力,1闪出料温度,2闪出料温度,3闪出料温度,4闪出料温度,5闪出料温度,5闪去循环母液调配槽密度,5闪去循环母液调配槽流量,新蒸汽进料压力,新蒸汽进料流量,新蒸汽进料温度,循环上水温度,循环上水流量,循环下水温度,原液闪蒸器循环母液进料阀开度,1效出料阀门开度,1闪出料阀开度,3闪出料阀开度,4闪出料阀开度,2效到1效泵开度,4效到3效泵开度,5效到4效泵开度,6效出料出料泵开度,1效蒸发器汽室压力,1效原液换热前温度,1效出料温度,1效蒸发器汽温'
    --target_cols '1效蒸发器汽温'  
    --seq_len 30 
    --label_len 2 
    --pred_len 30 
    --timestamp_feature none  
    --train_epochs 50 
    --gpu 0 
    --use_nni
    --checkpoints ./checkpoints/alumina_timexer_train_only
    --hidden_dim 128
    --d_model 128
    --e_layers 1
    --d_layers 1
    '''
echo $cmd1
$cmd1

cmd2='''python -u main_nni.py 
    --stage train_only 
    --model_id alumina_timexer
    --model_name alumina_timexer
    --features MS
    --log_path log 
    --trainset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_trainset.csv 
    --validset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_validset.csv
    --testset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_testset_new.csv 
    --datetime_col dtime 
    --feature_cols '5效5闪到调配槽后循环母液Nk,5效5闪到调配槽后循环母液ak/Rp,6效出料密度,5闪出料密度,57效循环母液进料温度,57效循环母液进料密度,冷凝器出汽压力,5效5闪蒸发母液去循环母液调配槽总流量,2效蒸发器汽温,2效蒸发器汽室压力,2效原液换热前温度,3效蒸发器汽温,3效蒸发器汽室压力,3效原液换热前温度,4效蒸发器汽温,4效蒸发器汽室压力,4效原液换热前温度,5效蒸发器汽温,5效蒸发器汽室压力,5效原液换热前温度,5效循环母液进料流量,6效蒸发器汽温,6效蒸发器汽室压力,6效原液换热前温度,原液闪蒸器循环母液进料流量,原液闪蒸器出料温度,原液闪蒸器蒸汽压力,7效原液换热前温度,7效蒸发器汽温,7效蒸发器汽室压力,1闪出料温度,2闪出料温度,3闪出料温度,4闪出料温度,5闪出料温度,5闪去循环母液调配槽密度,5闪去循环母液调配槽流量,新蒸汽进料压力,新蒸汽进料流量,新蒸汽进料温度,循环上水温度,循环上水流量,循环下水温度,原液闪蒸器循环母液进料阀开度,1效出料阀门开度,1闪出料阀开度,3闪出料阀开度,4闪出料阀开度,2效到1效泵开度,4效到3效泵开度,5效到4效泵开度,6效出料出料泵开度,1效蒸发器汽温,1效原液换热前温度,1效出料温度,1效蒸发器汽室压力'
    --target_cols '1效蒸发器汽室压力'  
    --seq_len 30 
    --label_len 2 
    --pred_len 30 
    --timestamp_feature none  
    --train_epochs 50 
    --gpu 0 
    --use_nni
    --checkpoints ./checkpoints/alumina_timexer_train_only
    --hidden_dim 128
    --d_model 128
    --e_layers 1
    --d_layers 1
    '''
echo $cmd2
$cmd2

cmd3='''python -u main_nni.py 
    --stage train_only 
    --model_id alumina_timexer
    --model_name alumina_timexer
    --features MS
    --log_path log 
    --trainset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_trainset.csv 
    --validset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_validset.csv
    --testset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_testset_new.csv 
    --datetime_col dtime 
    --feature_cols '5效5闪到调配槽后循环母液Nk,5效5闪到调配槽后循环母液ak/Rp,6效出料密度,5闪出料密度,57效循环母液进料温度,57效循环母液进料密度,冷凝器出汽压力,5效5闪蒸发母液去循环母液调配槽总流量,2效蒸发器汽温,2效蒸发器汽室压力,2效原液换热前温度,3效蒸发器汽温,3效蒸发器汽室压力,3效原液换热前温度,4效蒸发器汽温,4效蒸发器汽室压力,4效原液换热前温度,5效蒸发器汽温,5效蒸发器汽室压力,5效原液换热前温度,5效循环母液进料流量,6效蒸发器汽温,6效蒸发器汽室压力,6效原液换热前温度,原液闪蒸器循环母液进料流量,原液闪蒸器出料温度,原液闪蒸器蒸汽压力,7效原液换热前温度,7效蒸发器汽温,7效蒸发器汽室压力,1闪出料温度,2闪出料温度,3闪出料温度,4闪出料温度,5闪出料温度,5闪去循环母液调配槽密度,5闪去循环母液调配槽流量,新蒸汽进料压力,新蒸汽进料流量,新蒸汽进料温度,循环上水温度,循环上水流量,循环下水温度,原液闪蒸器循环母液进料阀开度,1效出料阀门开度,1闪出料阀开度,3闪出料阀开度,4闪出料阀开度,2效到1效泵开度,4效到3效泵开度,5效到4效泵开度,6效出料出料泵开度,1效蒸发器汽温,1效蒸发器汽室压力,1效出料温度,1效原液换热前温度'
    --target_cols '1效原液换热前温度'  
    --seq_len 30 
    --label_len 2 
    --pred_len 30 
    --timestamp_feature none  
    --train_epochs 50 
    --gpu 0 
    --use_nni
    --checkpoints ./checkpoints/alumina_timexer_train_only
    --hidden_dim 128
    --d_model 128
    --e_layers 1
    --d_layers 1
    '''
echo $cmd3
$cmd3

cmd4='''python -u main_nni.py 
    --stage train_only 
    --model_id alumina_timexer
    --model_name alumina_timexer
    --features MS
    --log_path log 
    --trainset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_trainset.csv 
    --validset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_validset.csv
    --testset_csv_path /workspace/qiuyunzhong/TSFrameworkData/alumina_testset_new.csv 
    --datetime_col dtime 
    --feature_cols '5效5闪到调配槽后循环母液Nk,5效5闪到调配槽后循环母液ak/Rp,6效出料密度,5闪出料密度,57效循环母液进料温度,57效循环母液进料密度,冷凝器出汽压力,5效5闪蒸发母液去循环母液调配槽总流量,2效蒸发器汽温,2效蒸发器汽室压力,2效原液换热前温度,3效蒸发器汽温,3效蒸发器汽室压力,3效原液换热前温度,4效蒸发器汽温,4效蒸发器汽室压力,4效原液换热前温度,5效蒸发器汽温,5效蒸发器汽室压力,5效原液换热前温度,5效循环母液进料流量,6效蒸发器汽温,6效蒸发器汽室压力,6效原液换热前温度,原液闪蒸器循环母液进料流量,原液闪蒸器出料温度,原液闪蒸器蒸汽压力,7效原液换热前温度,7效蒸发器汽温,7效蒸发器汽室压力,1闪出料温度,2闪出料温度,3闪出料温度,4闪出料温度,5闪出料温度,5闪去循环母液调配槽密度,5闪去循环母液调配槽流量,新蒸汽进料压力,新蒸汽进料流量,新蒸汽进料温度,循环上水温度,循环上水流量,循环下水温度,原液闪蒸器循环母液进料阀开度,1效出料阀门开度,1闪出料阀开度,3闪出料阀开度,4闪出料阀开度,2效到1效泵开度,4效到3效泵开度,5效到4效泵开度,6效出料出料泵开度,1效蒸发器汽温,1效蒸发器汽室压力,1效原液换热前温度,1效出料温度'
    --target_cols '1效出料温度'  
    --seq_len 30 
    --label_len 2 
    --pred_len 30 
    --timestamp_feature none  
    --train_epochs 50 
    --gpu 0 
    --use_nni
    --checkpoints ./checkpoints/alumina_timexer_train_only
    --hidden_dim 128
    --d_model 128
    --e_layers 1
    --d_layers 1
    '''
echo $cmd4
$cmd4