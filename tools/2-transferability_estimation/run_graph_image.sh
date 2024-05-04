echo 
echo ====== new command ========
echo `realpath .`
for CONTAIN_DATASET_FEATURE in True; #  True  False
do
    echo CONTAIN_DATASET_FEATURE-$CONTAIN_DATASET_FEATURE
    for CONTAIN_DATA_SIMILARITY in True; #True
    do
        echo CONTAIN_DATA_SIMILARITY-$CONTAIN_DATA_SIMILARITY
        task_type=image_classification
        for dataset in cifar100,caltech101,dtd,flowers,pets,smallnorb_label_elevation,stanfordcars,svhn;
        do
            echo dataset-$dataset
            for complete_model_features in True; # False
            do
                echo complete_model_features-${complete_model_features}
                for top_pos_K in 0.5; # .5
                do
                    echo top_pos_K-$top_pos_K
                    for top_neg_K in 0.5; #0.5
                    do
                        echo top_neg_K-$top_neg_K
                        for accu_neg_thres in 0.5; #0.5
                        do 
                            echo accu_neg_thres-$accu_neg_thres
                            for dataset_embed_method in domain_similarity; #; #task2vec
                            do
                                for CONTAIN_MODEL_FEATURE in False; # False True
                                do
                                    echo CONTAIN_MODEL_FEATURE-$CONTAIN_MODEL_FEATURE
                                    for accu_pos_thres in 0.5; #0.5
                                    do 
                                        echo accu_pos_thres-$accu_pos_thres
                                        for hidden_channels in 128; #64 32
                                        do
                                            echo hidden_channels-$hidden_channels
                                            for GNN_METHOD in xgb_homoGATConv_all_normalize_without_transfer xgb_homo_SAGEConv_all_normalize_without_transfer xgb_node2vec_all_normalize_without_transfer xgb_node2vec+_all_normalize_without_transfer\
                                                              lr_homoGATConv_all_normalize_without_transfer lr_homo_SAGEConv_all_normalize_without_transfer lr_node2vec_all_normalize_without_transfer lr_node2vec+_all_normalize_without_transfer\
                                                              rf_homoGATConv_all_normalize_without_transfer rf_homo_SAGEConv_all_normalize_without_transfer rf_node2vec_all_normalize_without_transfer rf_node2vec+_all_normalize_without_transfer;
                                            do
                                                echo GNN_METHOD-$GNN_METHOD
                                                for finetune_ratio in 1.0; #1.0; #; # 0.3 0.7 0.5 
                                                do
                                                    echo "python3 tools/2-transferability_estimation/run.py"
                                                    echo "        --contain_data_similarity ${CONTAIN_DATA_SIMILARITY}"
                                                    echo "        --contain_dataset_feature ${CONTAIN_DATASET_FEATURE}"
                                                    echo "        --contain_model_feature ${CONTAIN_MODEL_FEATURE}"
                                                    echo "        --complete_model_features ${complete_model_features}"
                                                    echo "        --gnn_method ${GNN_METHOD}"
                                                    echo "        --test_dataset ${dataset}"
                                                    echo "        --top_neg_K ${top_neg_K}"
                                                    echo "        --top_pos_K ${top_pos_K}"
                                                    echo "        --accu_neg_thres ${accu_neg_thres}"
                                                    echo "        --accu_pos_thres ${accu_pos_thres}"
                                                    echo "        --hidden_channels ${hidden_channels}"
                                                    echo "        --finetune_ratio ${finetune_ratio}"
                                                    echo "        --dataset_embed_method ${dataset_embed_method}"
                                                    echo "        --task_type ${task_type}"
                                                    python3 tools/2-transferability_estimation/run.py \
                                                            --contain_data_similarity ${CONTAIN_DATA_SIMILARITY} \
                                                            --contain_dataset_feature ${CONTAIN_DATASET_FEATURE} \
                                                            --contain_model_feature ${CONTAIN_MODEL_FEATURE} \
                                                            --complete_model_features ${complete_model_features} \
                                                            --gnn_method ${GNN_METHOD} \
                                                            --test_dataset ${dataset} \
                                                            --top_neg_K ${top_neg_K} \
                                                            --top_pos_K ${top_pos_K} \
                                                            --accu_neg_thres ${accu_neg_thres} \
                                                            --accu_pos_thres ${accu_pos_thres} \
                                                            --hidden_channels ${hidden_channels} \
                                                            --finetune_ratio ${finetune_ratio} \
                                                            --dataset_embed_method ${dataset_embed_method} \
                                                            --task_type ${task_type}
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
