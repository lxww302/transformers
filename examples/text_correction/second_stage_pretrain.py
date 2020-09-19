from transformers import *
import torch


class SecondStagePretrain(object):

    def __init__(self, model_name, train_data_path, dev_data_path, output_data_path):
        self.output_data_path = output_data_path
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_dataset = LineByLineTextDataset(tokenizer=self.tokenizer, file_path=train_data_path, block_size=128)
        self.eval_dataset = LineByLineTextDataset(tokenizer=self.tokenizer, file_path=dev_data_path, block_size=128)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        self.training_args = TrainingArguments(
            output_dir=self.output_data_path,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            num_train_epochs=1,
            save_total_limit=10,

        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model()

    def eval(self):
        torch.cuda.empty_cache()
        eval_output = self.trainer.evaluate()
        print(eval_output)


if __name__ == '__main__':
    second = SecondStagePretrain(model_name='hfl/chinese-electra-large-generator',
                                 train_data_path='/mnt/nlp-lq/wuwei.ai/data/correction/hpi_train.txt',
                                 dev_data_path='/mnt/nlp-lq/wuwei.ai/data/correction/hpi_train.txt',
                                 output_data_path='/mnt/nlp-lq/wuwei.ai/data/correction/output')
    second.train()
    # second.eval()
