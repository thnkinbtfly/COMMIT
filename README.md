# COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning

Implemented based on [EasyLM](https://github.com/young-geng/EasyLM) and [lm-evaluation-harness](https://github.com/OpenGPTX/lm-evaluation-harness).

## Specialized Code-mixing for Instruction Tuning 
First, prepare the weight of LLaMA by following [EasyLM doc](docs/llama.md), and download the [MUSE](https://github.com/facebookresearch/MUSE) dictionary to `dicts`.

Then perform COMMIT on TPUv3-8 and convert to hf format as follows:
```bash
export save_path=SAVE_PATH
export lang=hi # or el, th
python -m EasyLM.models.llama.llama_train \
--mesh_dim=1,1,-1 \
--load_llama_config=7b \
--load_checkpoint=params::7B \
--total_steps=1210 \
--logger.output_dir=$save_path \
--save_milestone_freq=1210 \
--tokenizer.vocab_file=tokenizer.model \
--tokenizer.add_bos_token=True \
--tokenizer.add_eos_token=True \
--optimizer.type=adamw \
--optimizer.accumulate_gradient_steps=1 \
--optimizer.adamw_optimizer.weight_decay=0.0 \
--optimizer.adamw_optimizer.lr=2e-5 \
--optimizer.adamw_optimizer.b1=0.9 \
--optimizer.adamw_optimizer.b2=0.999 \
--optimizer.adamw_optimizer.end_lr=1e-7 \
--optimizer.adamw_optimizer.lr_warmup_steps=36 \
--optimizer.adamw_optimizer.lr_decay_steps=1210 \
--optimizer.adamw_optimizer.bf16_momentum=True \
--train_dataset.text_processor.alpaca=True \
--train_dataset.text_processor.codemix_dict_path=dicts/en-${lang}.txt \
--train_dataset.text_processor.codemix_ratio=0.9 \
--train_dataset.text_processor.block_codemix_in_template=True \
--train_dataset.type=huggingface \
--train_dataset.huggingface_dataset.path=tatsu-lab/alpaca \
--train_dataset.huggingface_dataset.seq_length=512 \
--train_dataset.huggingface_dataset.batch_size=16 \
--update_llama_config='{"resid_pdrop": 0.05, "embd_pdrop": 0.05, "attn_pdrop": 0.05, "fcm_max_ratio": 0.1}' --log_freq=5000

export hf_save_dir=HF_SAVE_DIR
python -m EasyLM.models.llama.convert_easylm_to_hf \
--load_checkpoint=params::$save_path/*/streaming_params_1210 \
--tokenizer_path=tokenizer.model \
--model_size='7b' \
--output_dir=$hf_save_dir
```

Finally, run evaluation on GPU
```bash
python main.py \
--model hf-causal-experimental \
--model_args pretrained=$hf_save_dir --batch_size=2 \
--tasks ogx_xquad_${lang} --output_path $hf_save_dir/result.json \
--device cuda

```