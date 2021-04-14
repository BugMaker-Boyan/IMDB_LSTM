

class Word2Sequence(object):
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.inverse_dict = {}
        self.count = {}

    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):

        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len-len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, sequence):
        return [self.inverse_dict.get(num) for num in sequence]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    # ws = Word2Sequence()
    # ws.fit(['我', '是', '谁', '我', '是', '我'])
    # ws.build_vocab(min=0)
    # print(ws.dict)
    # print(ws.inverse_dict)
    # ret = ws.transform(['我', '爱', 'Python'], max_len=10)
    # print(ret)
    # ret = ws.inverse_transform(ret)
    # print(ret)

    import pickle
    from word_sequence import Word2Sequence
    import os
    from dataset import tokenize
    from tqdm import tqdm

    ws = Word2Sequence()
    path = r'..\data\IMDB\train'
    temp_data_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    for data_path in temp_data_path:
        file_names = os.listdir(data_path)
        file_paths = [os.path.join(data_path, i) for i in file_names if i.endswith('.txt')]
        for file_path in tqdm(file_paths):
            sentence = tokenize(open(file_path, encoding='utf-8').read())
            ws.fit(sentence)
    ws.build_vocab(min=10)
    pickle.dump(ws, open("./model/ws.pkl", 'wb'))
