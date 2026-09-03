#!/bin/bash
set -e
echo "Cleaning up AI-X-DL-Projects repo..."

cat > "README.md" << 'PYEOF_MARKER'
# Anomaly Classification of Machine Acoustic Data Using Deep Learning
# 深層学習による機械音響データの異常状態分類
# 딥러닝을 이용한 기계장비 음향 데이터 기반 이상상태 분류

A CRNN (CNN + LSTM) model that listens to a machine's operating sound (converted to a
Mel-spectrogram) and classifies it as normal or abnormal — a predictive-maintenance
style anomaly detector using only audio, no vibration/thermal sensors.

기계 장비가 작동하며 내는 소리(Mel-spectrogram으로 변환)를 CRNN(CNN+LSTM) 모델에
입력해 정상/이상 상태를 분류하는 프로젝트입니다. 진동·열 센서 없이 오디오만으로
예지보전(predictive maintenance)형 이상 감지를 시도했습니다.

機械設備の稼働音(Mel-spectrogramに変換)をCRNN(CNN+LSTM)モデルに入力し、正常/
異常状態を分類するプロジェクトです。振動・熱センサーを使わず、音声のみで
予知保全(predictive maintenance)型の異常検知を試みました。

## Background / 背景 / 배경

This was done for **AI+X 딥러닝 (AI+X Deep Learning)**, an elective course taken in
the 2025-1 semester — the same semester as the mechanical engineering capstone
(`Recyclable-Waste-Sorting-System`) — after getting interested in machine learning
through that capstone work. Team: 임규원, 이재룡 (Option A).

이 프로젝트는 **AI+X 딥러닝**이라는 선택 과목에서 진행했습니다. 2025-1학기,
기계공학 종합설계1(재활용품 분류 프로젝트)을 진행하던 것과 같은 학기에 그
프로젝트를 계기로 머신러닝에 흥미가 생겨서 수강했습니다. 팀: 임규원, 이재룡
(Option A).

本プロジェクトは選択科目**「AI+X 딥러닝(AI+X ディープラーニング)」**で行いました。
2025年度1学期、機械工学総合設計1(リサイクル品分類プロジェクト)を進めていたのと
同じ学期に、そのプロジェクトをきっかけに機械学習に興味を持ち履修しました。
チーム:イム・ギュウォン、イ・ジェリョン(Option A)。

## My role in the project / このプロジェクトでの担当 / 이 프로젝트에서 내 역할

임규원 handled essentially all of the programming: data preprocessing, feature
extraction (Mel-spectrogram), the CRNN model implementation and training, and the
evaluation code. 이재룡 handled the research/topic selection and direction-setting
(deciding on the acoustic-anomaly-detection framing and which datasets/techniques to
consider), plus the report writing and demo video recording.

임규원이 프로그래밍 전반(데이터 전처리, 특징 추출(Mel-spectrogram), CRNN 모델
구현 및 학습, 평가 코드)을 담당했습니다. 이재룡은 자료 조사·주제 선정·방향
정리(음향 기반 이상 탐지로 방향을 잡고 어떤 데이터셋/기법을 검토할지 결정)와
보고서 작성, 시연 영상 녹화를 담당했습니다.

イム・ギュウォンがプログラミング全般(データ前処理、特徴抽出(Mel-spectrogram)、
CRNNモデルの実装・学習、評価コード)を担当しました。イ・ジェリョンは資料調査・
テーマ選定・方向性の整理(音響ベースの異常検知という方向性を決め、どの
データセット・手法を検討するかを決定)、レポート作成、デモ動画の撮影を
担当しました。

## Problem & approach (from the README's original writeup) / 課題とアプローチ / 문제 정의 및 접근

Sensor-based vibration/thermal monitoring for predictive maintenance is accurate but
expensive to install and hard to apply broadly. Acoustic sensors are cheap, easy to
install, and non-contact — but acoustic data is noisy and unstructured, so it needs
careful feature extraction and a learned classifier rather than simple thresholding.
The goal was to build and validate an anomaly classifier for machine sound.

센서 기반 진동/열 모니터링은 정확하지만 설치 비용이 크고 범용적으로 적용하기
어렵습니다. 음향 센서는 저렴하고 설치가 간편하며 비접촉이라는 장점이 있지만,
음향 데이터는 노이즈가 많고 비정형적이라 정교한 특징 추출과 학습 기반 분류기가
필요합니다(단순 임계값 방식으로는 부족). 목표는 기계 음향 데이터를 이용한 이상
상태 분류 시스템을 만들고 실험적으로 타당성을 검증하는 것이었습니다.

センサーベースの振動・熱モニタリングは高精度だが、設置コストが大きく汎用的な
適用が難しい。音響センサーは安価で設置が容易、かつ非接触であるという利点が
あるが、音響データはノイズが多く非定型であるため、精緻な特徴抽出と学習ベースの
分類器が必要(単純な閾値方式では不十分)。目標は、機械音響データを用いた異常
状態分類システムを構築し、実験的にその妥当性を検証することだった。

## Method / 手法 / 방법

- **Dataset**: MIMII Dataset (Hitachi) — fan subset, normal/abnormal machine sounds.
- **Feature extraction**: Mel-spectrogram (sampling rate 16,000Hz, 64 Mel bins),
  converted to a 2D image-like array as model input.
- **Model**: CRNN (Convolutional Recurrent Neural Network) — CNN layers extract
  frequency-pattern features, an LSTM layer captures temporal changes, a final Dense
  layer classifies normal vs. abnormal.
- **Training**: CrossEntropyLoss, Adam (lr=0.001), 30 epochs. Loss dropped from ~98.6
  to ~3.3 with no clear overfitting signs (see `images/epoch_loss.png`).
- **Additional evaluation**: tested against the *train* split of a separate Kaggle
  dataset ("Anomaly Detection from Sound Data (Fan)") as a held-out check — 6,521
  clips, classified 75.0% normal / 25.0% abnormal (see `images/evaluation_result.png`).

- **데이터셋**: MIMII Dataset(Hitachi)의 fan 서브셋 — 정상/이상 기계음.
- **특징 추출**: Mel-spectrogram(샘플링레이트 16,000Hz, 64 Mel bins)을 2D 배열
  형태로 변환해 모델 입력으로 사용.
- **모델**: CRNN(CNN+LSTM) — CNN 레이어가 주파수 패턴 특징을 추출하고, LSTM
  레이어가 시간적 변화를 반영하며, 마지막 Dense 레이어가 정상/이상을 분류.
- **학습**: CrossEntropyLoss, Adam(lr=0.001), 30 epoch. Loss가 약 98.6에서 약
  3.3까지 감소했고 뚜렷한 과적합 징후는 없었습니다(`images/epoch_loss.png` 참고).
- **추가 평가**: 별도의 Kaggle 데이터셋("Anomaly Detection from Sound Data (Fan)")의
  train 스플릿을 홀드아웃 테스트로 사용 — 6,521개 클립 중 정상 75.0% / 이상
  25.0%로 분류(`images/evaluation_result.png` 참고).

- **データセット**:MIMII Dataset(Hitachi)のfanサブセット — 正常/異常な機械音。
- **特徴抽出**:Mel-spectrogram(サンプリングレート16,000Hz、64 Mel bins)を2D
  配列形式に変換しモデル入力として使用。
- **モデル**:CRNN(CNN+LSTM) — CNN層が周波数パターンの特徴を抽出し、LSTM層が
  時間的変化を捉え、最後のDense層で正常/異常を分類。
- **学習**:CrossEntropyLoss、Adam(lr=0.001)、30エポック。Lossは約98.6から約
  3.3まで低下し、明確な過学習の兆候はなかった(`images/epoch_loss.png`参照)。
- **追加評価**:別のKaggleデータセット("Anomaly Detection from Sound Data (Fan)")
  のtrain分割をホールドアウトテストとして使用 — 6,521クリップ中、正常75.0%/
  異常25.0%と分類(`images/evaluation_result.png`参照)。

## Why the accuracy is lower than hoped, and what's next
## 期待より精度が低い理由と今後の方向
## 정확도가 기대보다 낮은 이유와 향후 방향

The original writeup attributes the gap to: limited/imbalanced training data, relying
on Mel-spectrogram alone without additional features, limited model complexity, and
insufficient hyperparameter tuning. Suggested next steps: Transformer/attention-based
models, lighter models for mobile/edge deployment, and more data augmentation +
tuning.

기존 작성 내용에서는 이 격차의 원인으로 학습 데이터 부족/불균형, Mel-spectrogram
외 추가 특징 미활용, 모델 구조·복잡도의 한계, 하이퍼파라미터 튜닝 부족을
꼽았습니다. 제안된 다음 단계: Transformer/Attention 기반 모델, 모바일/엣지
디바이스용 경량 모델, 더 많은 데이터 증강과 튜닝.

元の記述では、このギャップの原因として学習データの不足・不均衡、
Mel-spectrogram以外の追加特徴量の未活用、モデル構造・複雑度の限界、
ハイパーパラメータチューニングの不足を挙げている。提案された次のステップ:
Transformer/Attentionベースのモデル、モバイル/エッジデバイス向けの軽量モデル、
より多くのデータ拡張とチューニング。

## What this repo contains / このリポジトリの内容 / 이 레포 구성

```
project.ipynb           Preprocessing, CRNN model, training, evaluation, and plots
crnn_mimii_fan.pth       Trained model weights
images/epoch_loss.png    Training-loss curve over 30 epochs
images/evaluation_result.png   Held-out evaluation result on the Kaggle test set
```

## Data & path notes / データとパスに関する注意 / 데이터·경로 관련 안내

The original notebook had hardcoded local paths (`D:\AI+XDL\dataset\fan`,
`D:/AI+XDL/archive/dev_data_fan/train`) from the author's own machine. These have been
replaced with relative paths — put the MIMII fan dataset under `data/fan/<id>/{normal,
abnormal}/*.wav` and the separate evaluation clips under `data/dev_data_fan/train/`,
or adjust the paths in `project.ipynb` to wherever you keep the data. Neither dataset
is included in this repo (not source, and MIMII especially is large).

기존 노트북에는 작성자 개인 컴퓨터의 로컬 경로(`D:\AI+XDL\dataset\fan`,
`D:/AI+XDL/archive/dev_data_fan/train`)가 하드코딩돼 있었습니다. 이걸 상대경로로
바꿨습니다 — MIMII fan 데이터셋은 `data/fan/<id>/{normal,abnormal}/*.wav`
구조로, 별도 평가용 클립은 `data/dev_data_fan/train/`에 두거나, `project.ipynb`
안의 경로를 본인이 데이터를 둔 위치로 바꿔서 쓰면 됩니다. 두 데이터셋 다 이
레포에는 포함돼 있지 않습니다(소스코드가 아니고, 특히 MIMII는 용량이 큼).

元のノートブックには、作成者個人のPCのローカルパス(`D:\AI+XDL\dataset\fan`、
`D:/AI+XDL/archive/dev_data_fan/train`)がハードコーディングされていました。
これらは相対パスに置き換えています — MIMII fanデータセットは
`data/fan/<id>/{normal,abnormal}/*.wav`の構成で、別途の評価用クリップは
`data/dev_data_fan/train/`に配置するか、`project.ipynb`内のパスをご自身の
データの場所に合わせて変更してください。どちらのデータセットも本リポジトリには
含まれていません(ソースコードではなく、特にMIMIIは容量が大きいため)。

## What changed from the original repo / 元のリポジトリからの変更点 / 원본 레포에서 바뀐 점

- Removed a duplicate cell: the notebook had two copies of `extract_mel()` back to
  back (an earlier draft, then the same function immediately followed by the actual
  data-loading loop). Kept only the complete version.
- Replaced hardcoded local paths with relative ones (see above).
- Removed decorative emoji from print statements and added Korean/Japanese/English
  comments throughout, consistent with the other repos in this cleanup series.
- Rewrote this README trilingually with fuller background/role context; content is
  otherwise the same as the original (course, dataset, method, results, limitations).

- 중복 셀 제거: 노트북에 `extract_mel()`이 연속으로 두 번 있었습니다(초기 초안
  하나와, 그 바로 뒤에 동일한 함수 + 실제 데이터 로딩 루프가 붙은 완성본).
  완성된 쪽만 남겼습니다.
- 하드코딩된 로컬 경로를 상대경로로 교체(위 내용 참고).
- print문의 장식용 이모지를 제거하고, 이번 정리 시리즈의 다른 레포들과 일관되게
  한국어/일본어/영어 주석을 추가했습니다.
- 이 README를 배경/역할 설명을 더 채워서 3개 언어로 다시 작성했습니다. 내용
  자체는 원본과 동일합니다(과목, 데이터셋, 방법론, 결과, 한계).

- 重複セルの削除:ノートブックに`extract_mel()`が連続して2回存在していました
  (初期草稿1つと、その直後に同一関数+実際のデータ読み込みループが続く完成版)。
  完成版のみを残しました。
- ハードコーディングされたローカルパスを相対パスに置き換え(上記参照)。
- print文の装飾的な絵文字を削除し、今回の整理シリーズの他のリポジトリと同様、
  韓国語・日本語・英語のコメントを全体に追加しました。
- このREADMEを、背景・役割説明をより充実させた上で3言語で書き直しました。
  内容自体は元の記述と同じです(科目、データセット、手法、結果、限界)。

## References / 参考資料 / 참고자료

- MIMII Dataset (Hitachi)
- Kaggle: [Anomaly Detection from Sound Data (Fan)](https://www.kaggle.com/datasets/vuppalaadithyasairam/anomaly-detection-from-sound-data-fan?resource=download)
- Amazon Lookout for Equipment — [official blog post](https://aws.amazon.com/ko/blogs/korea/acoustic-anomaly-detection-using-amazon-lookout-for-equipment/)
- Python libraries: Librosa, SciPy, PyTorch

PYEOF_MARKER

cat > "project.ipynb" << 'PYEOF_MARKER'
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import librosa\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_mel(file_path, sr=16000, n_mels=64):\n",
    "    y, _ = librosa.load(file_path, sr=sr)\n",
    "    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)\n",
    "    mel_db = librosa.power_to_db(mel, ref=np.max)\n",
    "    mel_norm = (mel_db + 40) / 40  # normalize: [-40, 0] dB -> [0, 1]\n",
    "    return mel_norm\n",
    "\n",
    "# fan 폴더의 상위 경로 (레포 루트에서 실행한다고 가정)\n",
    "# fanフォルダの親ディレクトリ(リポジトリのルートから実行する前提)\n",
    "# parent directory of the \"fan\" folder (assumes running from the repo root)\n",
    "root_dir = \"data/fan\"\n",
    "\n",
    "mel_data_array = []\n",
    "label_array = []\n",
    "\n",
    "for machine_id in os.listdir(root_dir):\n",
    "    id_path = os.path.join(root_dir, machine_id)\n",
    "    if not os.path.isdir(id_path):\n",
    "        continue\n",
    "\n",
    "    for status in ['normal', 'abnormal']:\n",
    "        status_path = os.path.join(id_path, status)\n",
    "        if not os.path.isdir(status_path):\n",
    "            continue\n",
    "\n",
    "        label = 0 if status == 'normal' else 1\n",
    "\n",
    "        for file in os.listdir(status_path):\n",
    "            if file.endswith(\".wav\"):\n",
    "                file_path = os.path.join(status_path, file)\n",
    "                mel = extract_mel(file_path)\n",
    "                mel_data_array.append(mel)\n",
    "                label_array.append(label)\n",
    "\n",
    "mel_data_array = np.array(mel_data_array)\n",
    "label_array = np.array(label_array)\n",
    "\n",
    "print(f\"Loaded {len(label_array)} samples\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "class MelSpectrogramDataset(Dataset):\n",
    "    def __init__(self, mel_list, label_list):\n",
    "        self.mel_list = mel_list\n",
    "        self.label_list = label_list\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.mel_list)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        mel = self.mel_list[idx]\n",
    "        label = self.label_list[idx]\n",
    "\n",
    "        mel_tensor = torch.tensor(mel, dtype=torch.float32)\n",
    "        label_tensor = torch.tensor(label, dtype=torch.long)\n",
    "\n",
    "        return mel_tensor, label_tensor\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = MelSpectrogramDataset(mel_data_array, label_array)\n",
    "dataloader = DataLoader(dataset, batch_size=32, shuffle=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CRNN(nn.Module):\n",
    "    def __init__(self, input_shape, n_classes):\n",
    "        super(CRNN, self).__init__()\n",
    "\n",
    "        self.cnn = nn.Sequential(\n",
    "            nn.Conv2d(1, 16, kernel_size=3, padding=1),\n",
    "            nn.BatchNorm2d(16),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2),\n",
    "\n",
    "            nn.Conv2d(16, 32, kernel_size=3, padding=1),\n",
    "            nn.BatchNorm2d(32),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2)\n",
    "        )\n",
    "\n",
    "        dummy_input = torch.zeros(1, 1, *input_shape)\n",
    "        cnn_out = self.cnn(dummy_input)\n",
    "        _, c, f, t = cnn_out.shape\n",
    "        self.rnn_input_size = f * c\n",
    "\n",
    "        self.rnn = nn.LSTM(self.rnn_input_size, 64, batch_first=True, bidirectional=True)\n",
    "\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Linear(64 * 2, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.3),\n",
    "            nn.Linear(64, n_classes)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.cnn(x)\n",
    "        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)\n",
    "        x = x.contiguous().view(x.shape[0], x.shape[1], -1)\n",
    "        x, _ = self.rnn(x)\n",
    "        x = x[:, -1, :]\n",
    "        return self.classifier(x)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = CRNN(input_shape=(64, mel_data_array.shape[2]), n_classes=2)\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "num_epochs = 30\n",
    "\n",
    "for epoch in range(num_epochs):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    for mel, label in dataloader:\n",
    "        mel = mel.unsqueeze(1)  # (B, 1, 64, T)\n",
    "        output = model(mel)\n",
    "        loss = criterion(output, label)\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        running_loss += loss.item()\n",
    "    print(f\"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "torch.save(model.state_dict(), \"crnn_mimii_fan.pth\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 모델 구조 먼저 다시 선언\n",
    "# まずモデル構造を再定義\n",
    "# Redeclare the model architecture first\n",
    "model = CRNN(input_shape=(64, mel_data_array.shape[2]), n_classes=2)\n",
    "model.load_state_dict(torch.load(\"crnn_mimii_fan.pth\"))\n",
    "model.eval()  # evaluation mode\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_audio(file_path, model, device=\"cpu\"):\n",
    "    mel = extract_mel(file_path)  # (64, T)\n",
    "    \n",
    "    if mel.shape[1] < 313:\n",
    "        # 패딩 (너무 짧은 경우)\n",
    "        # パディング(短すぎる場合)\n",
    "        # pad if too short\n",
    "        pad_width = 313 - mel.shape[1]\n",
    "        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')\n",
    "    elif mel.shape[1] > 313:\n",
    "        # 자르기 (너무 긴 경우)\n",
    "        # トリミング(長すぎる場合)\n",
    "        # trim if too long\n",
    "        mel = mel[:, :313]\n",
    "        \n",
    "    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 313)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        output = model(mel_tensor.to(device))\n",
    "        pred = torch.argmax(output, dim=1).item()\n",
    "        prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()\n",
    "\n",
    "    return pred, prob\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "normal_count = 0\n",
    "abnormal_count = 0\n",
    "total = 0\n",
    "\n",
    "# 레포 루트에서 실행한다고 가정한 상대경로\n",
    "# リポジトリのルートから実行する前提の相対パス\n",
    "# relative path, assumes running from the repo root\n",
    "target_folder = \"data/dev_data_fan/train\"\n",
    "\n",
    "for filename in os.listdir(target_folder):\n",
    "    if filename.endswith(\".wav\"):\n",
    "        path = os.path.join(target_folder, filename)\n",
    "        try:\n",
    "            pred, prob = predict_audio(path, model)\n",
    "            label = \"정상\" if pred == 0 else \"이상\"\n",
    "            print(f\"{filename}: {label} (정상확률: {prob[0]:.3f}, 이상확률: {prob[1]:.3f})\")\n",
    "\n",
    "            if pred == 0:\n",
    "                normal_count += 1\n",
    "            else:\n",
    "                abnormal_count += 1\n",
    "            total += 1\n",
    "        except Exception as e:\n",
    "            print(f\"[오류] {filename}: {e}\")\n",
    "\n",
    "# 예측 결과 통계 출력\n",
    "# 予測結果の統計を出力\n",
    "# print the prediction summary\n",
    "if total > 0:\n",
    "    print(\"\\n예측 결과 요약:\")\n",
    "    print(f\"정상: {normal_count}개 ({normal_count / total * 100:.1f}%)\")\n",
    "    print(f\"이상: {abnormal_count}개 ({abnormal_count / total * 100:.1f}%)\")\n",
    "    print(f\"총 예측 수: {total}개\")\n",
    "else:\n",
    "    print(\"예측된 오디오가 없습니다.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs = range(1, 31)\n",
    "loss = [\n",
    "    98.6, 91.3, 80.5, 72.5, 64.7, 52.7, 45.0, 41.8, 38.0, 32.0,\n",
    "    30.9, 26.8, 23.6, 19.5, 18.2, 19.0, 15.7, 12.7, 11.4, 12.1,\n",
    "    8.9, 9.1, 6.2, 6.8, 18.2, 6.1346, 3.5, 3.2, 7.9, 3.3\n",
    "]\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(epochs, loss, marker='o', linestyle='-', color='b', label='Training Loss')\n",
    "\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.title('Training Loss per Epoch (CRNN Model)')\n",
    "plt.xticks(epochs)\n",
    "plt.grid(True, linestyle='--')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0382a3c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import librosa.display\n",
    "\n",
    "# 레포 루트에서 실행한다고 가정한 상대경로\n",
    "# リポジトリのルートから実行する前提の相対パス\n",
    "# relative path, assumes running from the repo root\n",
    "audio_path = \"data/dev_data_fan/train/normal_id_00_00000000.wav\"\n",
    "y, sr = librosa.load(audio_path, sr=16000)\n",
    "mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_mels=64)\n",
    "mel_db = librosa.power_to_db(mel_spec, ref=np.max)\n",
    "\n",
    "plt.figure(figsize=(10, 4))\n",
    "librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')\n",
    "plt.colorbar(format='%+2.0f dB')\n",
    "plt.title('Mel-Spectrogram Example')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
PYEOF_MARKER

echo "Done. Run: git add -A && git commit -m 'cleanup: dedupe cell, relative paths, trilingual README' && git push"