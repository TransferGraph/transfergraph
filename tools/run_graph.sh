echo 
echo ====== new command ========
for CONTAIN_DATASET_FEATURE in True; #  True  False
do
    echo CONTAIN_DATASET_FEATURE-$CONTAIN_DATASET_FEATURE
    for CONTAIN_DATA_SIMILARITY in True; #True
    do
        echo CONTAIN_DATA_SIMILARITY-$CONTAIN_DATA_SIMILARITY
        modality=text
        for dataset in glue/sst2
        do
            # tweet_eval/sentiment  tweet_eval/emotion rotten_tomatoes glue/cola tweet_eval/irony tweet_eval/hate tweet_eval/offensive ag_news
            # cifar100 svhn stanfordcars dtd caltech101 smallnorb_label_elevation
            #  smallnorb_label_azimuth eurosat diabetic_retinopathy_detection 
            #   kitti   oxford_iiit_pet oxford_flowers102
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
                                            for GNN_METHOD in lr_homoGATConv lr_homoGCNConv lr_homo_SAGEConv lr_node2vec lr_node2vec+ 
                                                # 
                                                # lr_node2vec_without_accuracy lr_node2vec+_without_accuracy lr_homo_SAGEConv_without_accuracy lr_homoGATConv_without_accuracy
                                                # lr_node2vec+_without_transfer lr_node2vec_without_transfer
                                                #                   node2vec+_w2v rf_node2vec+ lr_node2vec+_all lr_node2vec_all
                                                #                   homo_SAGEConv_e2e homo_GATConv_e2e  homo_GCNConv_e2e
                                                #                   rf_node2vec_without_accuracy rf_node2vec+_without_accuracy
                                                #                   
                                                #                   lr_homo_GATConv_without_accuracy \ homo_SAGEConv_trained_on_transfer
                                                #                   lr_homo_SAGEConv_without_accuracy;
                                                #                   \    lr_homo_SAGEConv_e2e lr_homo_GATConv_e2e lr_homo_GCNConv_e2e
                                                #                   GATConv GATConv_trained_on_transfer GATConv_without_transfer \
                                                #                   node2vec SAGEConv  GATConv_without_transfer HeteroGNN  HGTConv  GATConv node2vec_without_transfer HeteroGNN SAGEConv node2vec; # #SAGEConv GATConv HGTConv; 
                                            do
                                                echo GNN_METHOD-$GNN_METHOD
                                                for finetune_ratio in 1.0; #1.0; #; # 0.3 0.7 0.5 
                                                do
                                                    echo "python3 run.py"
                                                    echo "        -contain_data_similarity ${CONTAIN_DATA_SIMILARITY}"
                                                    echo "        contain_dataset_feature ${CONTAIN_DATASET_FEATURE}"
                                                    echo "        -contain_model_feature ${CONTAIN_MODEL_FEATURE}"
                                                    echo "        -complete_model_features ${complete_model_features}"
                                                    echo "        -gnn_method ${GNN_METHOD}"
                                                    echo "        -test_dataset ${dataset}"
                                                    echo "        -top_neg_K ${top_neg_K}"
                                                    echo "        -top_pos_K ${top_pos_K}"
                                                    echo "        -accu_neg_thres ${accu_neg_thres}"
                                                    echo "        -accu_pos_thres ${accu_pos_thres}"
                                                    echo "        -hidden_channels ${hidden_channels}"
                                                    echo "        -finetune_ratio ${finetune_ratio}"
                                                    echo "        -dataset_embed_method ${dataset_embed_method}"
                                                    echo "        -modality ${modality}"
                                                    python3 run.py \
                                                            -contain_data_similarity ${CONTAIN_DATA_SIMILARITY} \
                                                            -contain_dataset_feature ${CONTAIN_DATASET_FEATURE} \
                                                            -contain_model_feature ${CONTAIN_MODEL_FEATURE} \
                                                            -complete_model_features ${complete_model_features} \
                                                            -gnn_method ${GNN_METHOD} \
                                                            -test_dataset ${dataset} \
                                                            -top_neg_K ${top_neg_K} \
                                                            -top_pos_K ${top_pos_K} \
                                                            -accu_neg_thres ${accu_neg_thres} \
                                                            -accu_pos_thres ${accu_pos_thres} \
                                                            -hidden_channels ${hidden_channels} \
                                                            -finetune_ratio ${finetune_ratio} \
                                                            -dataset_embed_method ${dataset_embed_method} \
                                                            -modality ${modality}                                                                                   ## -embed_dataset_feature ${EMBED_DATASET_FEATURE} \
                                                    ## -embed_model_feature ${EMBED_MODEL_FEATURE} \
                                                    ## -accuracy_thres ${ACCU_THRES} \
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
