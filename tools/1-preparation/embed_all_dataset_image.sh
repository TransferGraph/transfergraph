for dataset in cifar100,caltech101,dtd,flowers,pets,smallnorb_label_elevation,stanfordcars,svhn;
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
      echo "    --task_type=image_classification \ "
      echo "    --model_name=microsoft/resnet-50 \ "
      echo "    --embedding_method=${embed_method} \ "
      echo "    ${argument_dataset_path} \ "
      echo "    ${argument_dataset_name} "

      python3 tools/1-preparation/embed_dataset.py \
          --task_typ=sequence_classification \
          --model_name=microsoft/resnet-50 \
          --embedding_method=${embed_method} \
          --batch_size=16 \
          ${argument_dataset_path} \
          ${argument_dataset_name}
    done
done

