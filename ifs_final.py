# =========================
# 1. 라이브러리 설치 및 임포트
# =========================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

from transformers import BertForSequenceClassification
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
from kobert_tokenizer import KoBERTTokenizer

!pip install konlpy
from konlpy.tag import Komoran
komoran = Komoran()

import numpy as np
!pip install lime
from lime.lime_text import LimeTextExplainer
import torch
from tqdm import tqdm  # Add this for progress bar

# =========================
# 2. 설정
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# =========================
# 3. 데이터 로딩
# =========================
# df = pd.read_csv('output.csv', encoding='cp949')
df1 = pd.read_csv('temp.csv')
df2 = pd.read_csv('final.csv')
df1 = df1[['merged text']]
df2 = df2[['label']]
# 두 데이터프레임을 열 방향으로 합치기 (column-wise)
df = pd.concat([df1, df2], axis=1)
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# =========================
# 4. TF-IDF 키워드 추출 및 BERT 기준 토큰화
# =========================
def clean_text(text):
    text = re.sub('[^\uAC00-\uD7A3\s]', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def extract_nouns(text):
    text = clean_text(text)
    nouns = komoran.nouns(text)
    return ' '.join([w for w in nouns if len(w) > 1])

train_texts = train_data['merged text'].astype(str).apply(extract_nouns).tolist()
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(train_texts)
tfidf_vocab = tfidf.get_feature_names_out()

model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name)

doc_top_keywords = []
k = 10
for i in range(tfidf_matrix.shape[0]):
    row = tfidf_matrix.getrow(i)
    row_coo = list(zip(row.indices, row.data))
    row_coo_sorted = sorted(row_coo, key=lambda x: x[1], reverse=True)
    top_k = row_coo_sorted[:k]
    keywords = [tfidf_vocab[idx] for idx, val in top_k]

    tokenized_keywords = set()
    for keyword in keywords:
        tokens = tokenizer.tokenize(keyword)
        tokens = [tok.replace("##", "") for tok in tokens]
        tokenized_keywords.update(tokens)

    doc_top_keywords.append(tokenized_keywords)
    print(f"TF-IDF 키워드 (Top {k}):\n", keywords)

# =========================
# 5. KoBERT 모델 로딩 (4-class 설정)
# =========================
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4, output_attentions=True)
model.to(device)

# =========================
# 6. 토크나이징 및 Dataset 구성
# =========================
max_len = 256
train_encodings = tokenizer(train_data['merged text'].tolist(), truncation=True, padding=True, max_length=max_len)
test_encodings = tokenizer(test_data['merged text'].tolist(), truncation=True, padding=True, max_length=max_len)

y_train = train_data['label'].values - 1  # 라벨이 1~4라고 가정
y_test = test_data['label'].values - 1

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, y_train)
test_dataset = MyDataset(test_encodings, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# =========================
# 7. 커스텀 Loss 함수 (가중치 + 2차 패널티)
# =========================
def custom_loss_fn(outputs, labels, input_ids, doc_top_keywords_batch, lambda_=1.0):
    ce_loss = nn.CrossEntropyLoss()(outputs.logits, labels)
    last_attn = outputs.attentions[-1]
    batch_size = input_ids.shape[0]
    att_loss = 0.0

    for b in range(batch_size):
        keywords = doc_top_keywords_batch[b]
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
        important_idxs = [i for i, tok in enumerate(decoded_tokens) if tok in keywords]
        if not important_idxs:
            continue
        att_b = last_attn[b].mean(dim=0)
        cls_idx = 0
        for idx in important_idxs:
            att_val = att_b[cls_idx, idx]
            att_loss += (1.0 - att_val) ** 2

    att_loss /= batch_size
    return ce_loss + lambda_ * att_loss

# =========================
# 8. 학습 루프
# =========================
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
lambda_ = 1.0
epochs = 30

model.train()
for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        batch_keywords = []
        for b in range(len(labels)):
            global_idx = batch_idx * train_loader.batch_size + b
            batch_keywords.append(doc_top_keywords[global_idx])

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        loss = custom_loss_fn(outputs, labels, input_ids, batch_keywords, lambda_)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss = {avg_loss:.4f}")

# =========================
# 9. 평가
# =========================
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

f1 = f1_score(all_labels, all_preds, average='weighted')
print("\nTest F1 Score (weighted):", f1)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Class 1", "Class 2", "Class 3", "Class 4"]))

all_labels_restored = [x+1 for x in all_labels]
all_preds_restored = [x+1 for x in all_preds]

for i in range(len(all_preds)):
    print(f"[{i+1:>2}] 실제 라벨: {all_labels_restored[i]}  →  예측 라벨: {all_preds_restored[i]}")
    

# =========================
# 10. LIME 적용 (샘플)
# =========================

# 11-1) LIME용 예측함수
def bart_predict_proba(texts, batch_size=4):
    """
    LIME이 호출하는 예측 함수.
    입력: text list (ex: ["문장1", "문장2", ...])
    출력: 각 문장에 대한 확률분포 (shape: [n, num_labels])
    """
    model.eval()
    all_probs = []
    for idx in range(0, len(texts), batch_size):
        batch_texts = texts[idx:idx+batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (batch_size, num_labels)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.extend(probs)

    return np.array(all_probs)

# 클래스명(또는 라벨명) 정의
class_names = ['LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4']  # 실제 라벨명에 맞춰 수정

# 11-2) LIME Explainer 준비
explainer = LimeTextExplainer(
    class_names=class_names
)

# 11-3) 실제 테스트 데이터 중 일부 샘플만 LIME으로 시각화
#       (예: 처음 3개 샘플만)
num_samples_to_explain = 20

print("\n=== LIME 시각화 예시 ===")
for i in range(num_samples_to_explain):
    sample_text = test_data['merged text'].iloc[i]
    true_label = y_test[i] + 1
    print(f"\n[Sample {i+1}] 실제 라벨: {true_label}")

    # LIME 적용
    explanation = explainer.explain_instance(
        sample_text,
        bart_predict_proba,
        num_features=10,      # 중요 단어 10개
        labels=(0, 1, 2, 3),  # 우리가 가진 4개 라벨(0~3)
        num_samples=200       # perturbation sample 개수
    )

    # 노트북 환경이라면 HTML로 시각화 가능
    # (print로 대신, 중요 단어 리스트 출력)
    explanation.show_in_notebook(text=True)

    # 만약 CLI 환경이라면 as_list() 형태로 출력
    print("LIME 상위 feature 리스트(각 label별) ↓")
    for label_idx in [0, 1, 2, 3]:
        print(f"  >> Label {class_names[label_idx]}:")
        for word, weight in explanation.as_list(label=label_idx):
            print(f"     {word} => {weight:.4f}")

    print("-" * 50)