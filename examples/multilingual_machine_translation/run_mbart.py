from transformers import MBartForConditionalGeneration, MBartTokenizer

model = MBartForConditionalGeneration.from_pretrained("/data00/wuwei.ai/code/transformers/examples/seq2seq/zhen_finetune_04/best_tfmr")
# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
article = "中国人民站起来了！"
batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], src_lang='zh_CN', tgt_lang='en_XX')
translated_tokens = model.generate(**batch)
translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(translation)
