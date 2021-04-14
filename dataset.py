from torch.utils.data import DataLoader, Dataset
import os
import re
from lib import max_len, BATCH_SIZE, ws, TEST_BATCH_SIZE
import torch


class ImdbDataset(Dataset):
    def __init__(self, train=True):
        self.train_data_path = r'../data/IMDB/train'
        self.test_data_path = r'../data/IMDB/test'
        data_path = self.train_data_path if train else self.test_data_path

        temp_data_path = [os.path.join(data_path, 'pos'), os.path.join(data_path, 'neg')]
        self.total_file_path = []  # all path of data
        for path in temp_data_path:
            file_names = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_names if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path = self.total_file_path[index]
        label_str = file_path.split('\\')[-2]
        label = 0 if label_str == 'neg' else 1
        tokens = tokenize(open(file_path, encoding='utf-8').read())
        return tokens, label

    def __len__(self):
        return len(self.total_file_path)


def tokenize(content):
    content = re.sub("<.*>", " ", content)
    filters = ['\t', '\n', '\x97', '\x96', '#', '$', '%', '&', '"', '\.', ':']
    content = re.sub("|".join(filters), " ", content)
    tokens = [i.strip().lower() for i in content.split(" ") if i.strip() != '']
    return tokens


def collate_fn(batch):
    """
    :param batch: ([tokens, labels], [tokens, labels])
    :return: ([token_nums, token_nums], (labels, labels))
    """
    contents, labels = list(zip(*batch))
    contents = [ws.transform(i, max_len=max_len) for i in contents]
    contents = torch.LongTensor(contents)
    labels = torch.LongTensor(labels)
    return contents, labels


def get_dataloader(train=True):
    imdb_dataset = ImdbDataset(train=train)
    batch_size = BATCH_SIZE if train else TEST_BATCH_SIZE
    data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    for idx, (x, target) in enumerate(get_dataloader()):
        print(idx)
        print(x)
        print(target)
        break
