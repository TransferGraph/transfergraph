for dataset in glue/sst2 tweet_eval/sentiment tweet_eval/emotion rotten_tomatoes glue/cola tweet_eval/irony tweet_eval/hate tweet_eval/offensive;
  do
    for embed_method in domain_similarity
    do
      # Split the dataset_name into two parts based on the slash
      IFS='/' read -r dataset_path dataset_name <<< "$dataset"

      # Check if the second part is empty and set it to a Python-friendly 'None' if it is
      if [[ -z $dataset_name ]]; then
        argument_dataset_name=''
      else
        argument_dataset_name="--dataset_name=${dataset_name}"
      fi
      argument_dataset_path="--dataset_path=${dataset_path}"
      echo "python3 tools/1-preparation/embed_dataset.py \ "
      echo "    --task_type=sequence_classification \ "
      echo "    --model_name=EleutherAI_gpt-neo-125m \ "
      echo "    --embedding_method=${embed_method} \ "
      echo "    ${argument_dataset_path} \ "
      echo "    ${argument_dataset_name} "

      python3 tools/1-preparation/embed_dataset.py \
          --task_typ=sequence_classification \
          --model_name=EleutherAI/gpt-neo-125m \
          --embedding_method=${embed_method} \
          --batch_size=16 \
          ${argument_dataset_path} \
          ${argument_dataset_name}
    done
done

