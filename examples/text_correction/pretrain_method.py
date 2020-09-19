import random
from transformers import *
from tqdm import tqdm
import torch


class PretrainEvaluation(object):

    def __init__(self, data_path: str, model_name: str):
        self.data_dict = self.load_dict()
        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eval_data = self.load_data(data_path)

    def load_data(self, data_path: str):
        all_indices = []
        all_input_ids = []
        current_tokens = []
        current_labels = []
        with open(data_path) as fi:
            for line in tqdm(fi.readlines()):
                line_data = line.strip().split('\t')
                if len(line_data) != 2:
                    continue
                input_data = line_data[0].replace(' ', '')[: 126]
                label_data = line_data[1].replace(' ', '')[: 126]
                if len(input_data) != len(label_data):
                    continue
                indices = [i for i, (inp, lab) in enumerate(zip(input_data, label_data)) if inp != lab]
                select_index = indices[0] if len(indices) == 1 else random.randint(0, len(input_data) - 1)
                input_ids = [self.tokenizer.tokenize(token)[0] for token in input_data]
                input_ids[select_index] = self.tokenizer.mask_token
                input_tokens = [self.tokenizer.cls_token] + input_ids + [self.tokenizer.sep_token] + (126 - len(input_ids)) * [self.tokenizer.pad_token]
                current_tokens.append(input_data[select_index])
                current_labels.append(label_data[select_index])
                all_input_ids.append(self.tokenizer.convert_tokens_to_ids(input_tokens))
                all_indices.append(select_index + 1)

        return all_indices, all_input_ids, current_tokens, current_labels

    def predict(self, all_indices, all_input_ids, batch_size=64, topk=100):
        predict_values = []
        predict_indices = []
        for batch_id in tqdm(range(len(all_input_ids) // batch_size + 1)):
            batch = all_input_ids[batch_id * batch_size: min(len(all_input_ids), (batch_id + 1) * batch_size)]
            batch_indices = all_indices[batch_id * batch_size: min(len(all_input_ids), (batch_id + 1) * batch_size)]
            batch_ = torch.tensor(batch).cuda()
            prediction = self.model(batch_)[0]
            selected_prediction = prediction[torch.arange(prediction.size(0)), torch.tensor(batch_indices)].cpu().detach()
            topk_values, topk_indices = torch.topk(selected_prediction, k=topk, dim=1)  # (64, 100)
            predict_values.extend(topk_values.softmax(-1).numpy().tolist())
            predict_indices.extend(topk_indices.numpy().tolist())
        return predict_values, predict_indices

    def evaluate(self, filter: bool = True, eval_at_k: int = 3):
        all_indices, all_input_ids, current_tokens, current_labels = self.eval_data
        prediction_values, prediction_indices = self.predict(all_indices=all_indices, all_input_ids=all_input_ids, batch_size=64)  # (21137, 100)
        correct_num = 0
        for current_token, current_label, prediction_index in zip(current_tokens, current_labels, prediction_indices):
            similar_tokens = self.data_dict.get(current_token, set())
            predict_tokens = self.tokenizer.convert_ids_to_tokens(prediction_index)
            if not filter:
                if current_label in predict_tokens[: eval_at_k]:
                    correct_num += 1
            else:
                candidates = [candidate for candidate in predict_tokens if candidate in similar_tokens]
                result = candidates + [candidate for candidate in predict_tokens if candidate not in similar_tokens]
                if current_label in result[: eval_at_k]:
                    correct_num += 1
        print(F"accuracy @ {eval_at_k} for model {self.model_name}: {correct_num/len(all_indices)}")

    def load_dict(self, same_stroke_path='/data00/wuwei.ai/data/correction/same_stroke.txt', similar_pinyin_path='/data00/wuwei.ai/data/correction/similar_pinyin.txt.txt'):
        char_dict = {}
        with open(same_stroke_path) as fi:
            for line in fi:
                chars = line.strip().split()
                for char in chars:
                    if char not in char_dict:
                        char_dict[char] = set()
                    for similar in chars:
                        char_dict[char].add(similar)
        with open(similar_pinyin_path) as fi:
            for line in fi:
                char = line[0]
                similar = set(line.replace('\n', '').replace('\t', '').replace(' ', ''))
                if char not in char_dict:
                    char_dict[char] = set()
                char_dict[char].update(similar)
        return char_dict


if __name__ == '__main__':
    model_names = ['hfl/chinese-electra-small-generator',
                   'bert-base-chinese',
                   'hfl/chinese-bert-wwm-ext',
                   'hfl/chinese-bert-wwm',
                   'hfl/chinese-roberta-wwm-ext-large',
                   'hfl/chinese-electra-base-generator',
                   'hfl/chinese-electra-large-generator',
                   'clue/roberta_chinese_base',
                   'clue/roberta_chinese_large',
                   'voidful/albert_chinese_large',
                   'voidful/albert_chinese_xlarge',
                   'voidful/albert_chinese_xxlarge',
                   'adamlin/bert-distil-chinese',
                   ]
    # for model_name in model_names:
    evaluation = PretrainEvaluation('/data00/wuwei.ai/data/correction/test.txt', 'hfl/chinese-electra-large-generator')
    evaluation.evaluate()
