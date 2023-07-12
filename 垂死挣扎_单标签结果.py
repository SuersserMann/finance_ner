import os
import json
from collections import Counter
from tqdm import tqdm


# 合并文件
def merge_files(folder_path):
    merged_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if len(merged_data) <= i:
                        merged_data.append(data)
                    else:
                        merged_data[i]['events'].extend(data['events'])

    return merged_data


# 创建新的数据，只保留出现次数大于等于9的事件
def filter_data(merged_data):
    new_data = []
    for record in merged_data:
        counter = Counter((str(event['type']), str(event['entity'])) for event in record['events'])
        new_events = [event for event in record['events'] if counter[(str(event['type']), str(event['entity']))] >= 2]
        new_data.append({
            'text_id': record['text_id'],
            'events': new_events
        })
    return new_data


# 去除 events 中重复的数据
def remove_duplicates(data):
    filtered_data = []

    for item in data:
        events = item.get('events', [])

        if events:
            unique_events = []
            unique_keys = set()

            for event in events:
                key = (event['type'], event['entity'])

                if key not in unique_keys:
                    unique_keys.add(key)
                    unique_events.append(event)

            item['events'] = unique_events

        filtered_data.append(item)

    return filtered_data


def main():
    # 合并文件
    merged_data = merge_files("多标签/")

    # 过滤数据
    filtered_data = filter_data(merged_data)

    # 去除重复的数据
    final_data = remove_duplicates(filtered_data)

    # 写入新的文件
    with open('result.txt', 'w', encoding='utf-8') as file:
        for item in tqdm(final_data, desc='Writing Data'):
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


# 执行主函数
if __name__ == "__main__":
    main()
