import random
import jsonlines
import torch.utils.data as data


class Dataset1(data.Dataset):
    def __init__(self, filename):
        with jsonlines.open(filename, 'r') as f:
            self.data = list(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        text_id = item['text_id']
        text = item['text']
        events = item['events']

        return {"text_id": text_id, "text": text, "events": events}


def filter_empty_events(dd):
    add_0_dataset = [item for item in dd if item["events"] == []]
    return add_0_dataset


b_train_dataset = Dataset1('train.jsonl')
b_val_dataset = Dataset1('dev.jsonl')

train_0_dataset = filter_empty_events(b_train_dataset)
val_0_dataset = filter_empty_events(b_val_dataset)

seed_list = [42, 4000, 50, 90, 123, 1100, 1600, 2023]
for seed in seed_list:
    random.seed(seed)
    random_train = random.sample(train_0_dataset, 10000)
    random_val = random.sample(val_0_dataset, 1000)

    train_dataset_file = f"random_train_{seed}.jsonl"
    val_dataset_file = f"random_val_{seed}.jsonl"

    with jsonlines.open(train_dataset_file, 'w') as f_train:
        for data in random_train:
            f_train.write(data)

    with jsonlines.open(val_dataset_file, 'w') as f_val:
        for data in random_val:
            f_val.write(data)

print("All jsonl files are created successfully.")
