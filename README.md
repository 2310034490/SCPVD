##Reveal
python run.py --output_dir=./saved_models --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_train --train_data_file=../dataset/data/reveal/train_reveal.jsonl --eval_data_file=../dataset/data/reveal/valid_reveal.jsonl --test_data_file=../dataset/data/reveal/test_reveal.jsonl --epoch 8 --block_size 400 --train_batch_size 5 --eval_batch_size 8 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --d_size 128 --gradient_accumulation_steps 8 --gpu_id 1 (--enable_best_model)

python run.py --output_dir=./saved_models --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_test  --train_data_file=../dataset/data/reveal/train_reveal.jsonl --eval_data_file=../dataset/data/reveal/valid_reveal.jsonl --test_data_file=../dataset/data/reveal/test_reveal.jsonl --gpu_id 1 --checkpoint_type=best

python ../evaluator/evaluator.py -a ../dataset/data/reveal/test_reveal.jsonl -p saved_models/predictions.txt




##Reveal (Chrome + Debian)
https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy
vulnerables.json + non-vulnerables.json -> reveal.json -> train_reveal.jsonl + valid_reveal.jsonl + test_reveal.jsonl

##Devign (FFmpeg+Qemu)
https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/edit
function.json -> devign.json -> train_devign.jsonl + valid_devign.jsonl + test_devign.jsonl

##BigVul
https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/blob/master/all_c_cpp_release2.0.csv
MSR_data_cleaned.csv -> MSR_data_cleaned.json -> train_bigvul.jsonl + valid_bigvul.jsonl + test_bigvul.jsonl
