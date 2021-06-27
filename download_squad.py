import json

def data_formatter(path, save = False):
    with open(path, "r") as f:
        data = json.load(f)['data']
    train_data = []
    for i in range(len(data)):
        ls ={}
        for j in range(len(data[i]['paragraphs'])):
            for k in range(len(data[i]['paragraphs'][j]['qas'])):
                if not data[i]['paragraphs'][j]['qas'][k]['is_impossible']:
                    ls['id'] = data[i]['paragraphs'][j]['qas'][k]['id']
                    ls['question'] = data[i]['paragraphs'][j]['qas'][k]['question']
                    ls['answer'] = data[i]['paragraphs'][j]['qas'][k]['answers'][0]['text']
                    train_data.append(ls)
                    ls ={}
    if save:
        with open('train_formatted.jsonl', 'w') as fp:
            json.dump(train_data, fp)


if __name__ =='__main__':
    data_formatter('train-v2.0.json', True)
        