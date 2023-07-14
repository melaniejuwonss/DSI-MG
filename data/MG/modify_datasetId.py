import json


def create_index():
    know2id = json.load(open('knowledge_kmeansid.json', 'r', encoding='utf-8'))
    all_knowledge = json.load(open('mgcrs_allknowledges.json', 'r', encoding='utf-8'))

    saveList = list()
    for knowledge in all_knowledge:
        know_text = knowledge
        orgIdx = all_knowledge[know_text]
        kmeansIdx = know2id[str(orgIdx)]

        saveList.append({'context_tokens': [know_text], 'item': kmeansIdx})

    with open('indexing.json', 'w', encoding='utf-8') as wf:
        wf.write(json.dumps(saveList, indent=4))


def create_traintest():
    know2id = json.load(open('knowledge_kmeansid.json', 'r', encoding='utf-8'))
    datasets = ['mgcrs_train_dataset.json','mgcrs_test_gold_idx.json']
    for dataset in datasets:
        saveList = list()
        orgData = json.load(open(dataset, 'r', encoding='utf-8'))
        # id2dialog = {v: k for k, v in orgData.items()}
        for dialog_data in orgData:
            for key, value in dialog_data.items():
                dialog = key
                orgIdx = value
                if orgIdx == 7015:
                    print()
                kmeansIdx = know2id[str(orgIdx)]

                saveList.append({'context_tokens': [dialog], 'item': kmeansIdx})
        with open('kmeans_' + dataset, 'w', encoding='utf-8') as wf:
            wf.write(json.dumps(saveList, indent=4))


def mergeDataset():
    result = list()
    filenames = ['train_dialog.json', f'indexing.json']
    for f1 in filenames:
        with open(f1, 'r') as infile:
            tmp_file = json.load(infile)
            result.extend(tmp_file)
    with open('train.json', 'w', encoding='utf-8') as wf:
        wf.write(json.dumps(result, indent=4))


def getOnlyTrainKnowledge():
    train_idx = json.load(open('mgcrs_train_knowledge_index.json', 'r', encoding='utf-8'))
    train_data = json.load(open('mgcrs_train_dataset.json', 'r', encoding='utf-8'))
    allKnowledge = json.load(open('mgcrs_allknowledges.json', 'r', encoding='utf-8'))
    know2id = json.load(open('knowledge_kmeansid.json', 'r', encoding='utf-8'))
    saveList = list()

    for knowledge in allKnowledge:
        know_text = knowledge
        orgIdx = allKnowledge[know_text]
        if orgIdx in train_idx:
            kmeansIdx = know2id[str(orgIdx)]
            saveList.append({'context_tokens': [know_text], 'item': kmeansIdx}) # 3258
    with open('indexing.json', 'w', encoding='utf-8') as wf:
        wf.write(json.dumps(saveList, indent=4))


create_traintest()
# create_index()
# getOnlyTrainKnowledge()
# mergeDataset()

