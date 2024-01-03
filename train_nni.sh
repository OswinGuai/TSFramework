


#nnictl create --config config_hyperband_210_transformer.yaml --port 8082
#nnictl create --config config_hyperband_210_itransformer.yaml --port 8083
#nnictl create --config config_hyperband_210_patchtst.yaml --port 8084


# nnictl create --config transformer_pretrain.yaml --port 8083
# nnictl create --config transformer_pretrain.yaml --port 8085
# nnictl create --config transformer_finetune.yaml --port 8086
# nnictl create --config transformer_finetune.yaml --port 8094
# nnictl create --config transformer_train.yaml --port 8087

nnictl create --config itransformer_pretrain.yaml --port 9000
