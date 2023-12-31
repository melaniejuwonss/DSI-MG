# DSI-MG

## 1. Dataset 설명 (directory: data > MG)
### indexing.json
* Pre-training 용 dataset 파일 (총 sample: 3,258)
* training 대상이 되는 {knowledge text : kmeans id} 로 구성
  
### test.json
* DSI test file (총 sample: 3,711)
* {dialog text : kmeans 채번 index}

### train.json
* DSI train file (총 sample: 14,879 -> 원래 train sample + indexing sample)
* {dialog (knowledge) text : kmeans 채번 index}

### knowledge_kmeansid.json
* {HJ 채번 index : kmeans 채번 index}

### knowledge_kmeans.json
* HJ가 채번한 knowledge index (1 ~ 10752) : kmeans 로 채번한 index

### mgcrs_allknowledges.json
* HJ 가 전달해준 모든 knowledges {knowledge text : HJ 채번 index}

### mgcrs_test_dataset.json
* HJ 가 전달해준 test sample (총 sample: 3,711)
* {text : HJ 채번 index}

### mgcrs_train_dataset.json
* HJ 가 전달해준 train sample (총 sample: 11,621)
* {text : HJ 채번 index}

### mgcrs_train_knowledge_index.json
* HJ 가 전달해준 indexing 대상이 되는 HJ 채번 index
* set 으로 바꾸면 총 3,258 개

### kmeans_mg.py
* Kmeans clustering 으로 id 채번하기 위한 파일
* Output file: knowledge_kmeansid.json
* 파일 실행시키기 위해서 kmeans_clustering library 설치 필요
``` pip install kmeans-pytorch ```

### modify_datasetId.py
* create_traintest()
  * HJ 가 준 train/test file 에 있는 index 를 kmeans index 로 변환
* create_index()
  * HJ 가 준 allknowledges 에 있는 index 를 kmeans index 로 변환
* getOnlyTrainKnowledge()
  * allknowledges 중 train 대상이 되는 것만 따로 저장 (최종 indexing.json)
* mergeDataset()
  * creat_traintest() 의 결과물 중 'train' 만 'indexing.json' 과 merge 작업 수행   

## 2. Train DSI-MG
### Command
``` python train.py --learning_rate=5e-4 --name=MG-t5Large-le5e4-noPre-dialogLen250-warmup --prefix=True  --num_train_epochs=15 --train_type=0 --max_dialog_len=250 --device=1; ```
### Result
* T5 가 예측한 id, label, input 저장 (results > MMYY_RecPred)
* Hit@K score 저장 (results > MMYY_raw)
