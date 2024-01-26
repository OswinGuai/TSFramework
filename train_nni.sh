


#nnictl create --config config_hyperband_210_transformer.yaml --port 8082
#nnictl create --config config_hyperband_210_itransformer.yaml --port 8083
#nnictl create --config config_hyperband_210_patchtst.yaml --port 8084


# nnictl create --config transformer_pretrain.yaml --port 8083
# nnictl create --config transformer_pretrain.yaml --port 8085
# nnictl create --config transformer_finetune.yaml --port 8086
# nnictl create --config transformer_finetune.yaml --port 8094
# nnictl create --config transformer_train.yaml --port 8087

# nnictl create --config itransformer_pretrain.yaml --port 9000
nnictl create --config itransformer_pretrain.yaml --port 9001

nnictl create --config itransformer_train_only.yaml --port 9010 # underfit
nnictl create --config itransformer_train_only.yaml --port 9011
nnictl create --config itransformer_train_only.yaml --port 9012 # multi embedding
nnictl create --config itransformer_train_only.yaml --port 9013 # multi embedding | high encoder layer
nnictl create --config itransformer_train_only.yaml --port 9014 # single embedding | high encoder layer

nnictl create --config itransformer_finetune.yaml --port 9020

nnictl create --config patchtst_pretrain.yaml --port 9101

nnictl create --config patchtst_train_only.yaml --port 9110 # underfit
nnictl create --config patchtst_train_only.yaml --port 9112

nnictl create --config patchtst_finetune.yaml --port 9120

nnictl create --config transformer_train_only.yaml --port 9210
nnictl create --config transformer_train_only.yaml --port 9211
nnictl create --config transformer_train_only.yaml --port 9212

nnictl create --config transformer_new_train_only.yaml --port 9220 # multi embedding | documented
nnictl create --config transformer_new_train_only.yaml --port 9221 # single embedding | documented
nnictl create --config transformer_new_train_only.yaml --port 9222 # encoder & decoder multi linear embedding | documented | need rerun
nnictl create --config transformer_new_train_only.yaml --port 9223 # encoder multi linear embedding | decoder single linear embedding |  documented
nnictl create --config transformer_new_train_only.yaml --port 9224 # encoder & decoder single linear embedding |  documented
nnictl create --config transformer_new_train_only.yaml --port 9225 # encoder & decoder multi linear embedding | no position | documented | need rerun
nnictl create --config transformer_new_train_only.yaml --port 9226 # encoder & decoder multi original embedding | not documented


nnictl create --config transformer_train_reg.yaml --port 9250
nnictl create --config transformer_train_reg.yaml --port 9251