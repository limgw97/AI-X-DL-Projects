# AI-X-DL-Projects

# 기계장비에서 발생하는 음향 데이터를 이용한 이상상태 분류

**조원**  
- 임규원 (기계공학부) - limgw97@hanyang.ac.kr
- 이재룡 (기계공학부) - aszx3263@naver.com  

(Option A 선택)

---

## 1. 프로젝트 개요 및 목표 설정

### 1.1 프로젝트 개요

최근 산업 현장에서는 다양한 기계장비가 고도의 자동화·지능화 과정을 거치며 사용되고 있으며, 이들 장비의 상태를 실시간으로 감지하고 이상 여부를 판단하는 기술의 중요성이 커지고 있습니다.  
특히 기계의 고장은 생산 라인 전체에 영향을 미치기 때문에 조기 진단 및 예방 유지보수(Predictive Maintenance)의 필요성이 강조되고 있습니다.

기계 장비는 작동 중 고유한 음향 신호를 발생시키며, 장비 내부의 마모, 고장, 정렬 불량 등 이상 상태가 발생할 경우 이러한 음향 패턴에 미세한 변화가 감지됩니다.  
**본 프로젝트는 이러한 음향 신호의 특성 변화를 분석하여 기계 장비의 정상/비정상 상태를 분류**하고자 합니다.

### 1.2 문제 정의

기존 센서 기반 진동 또는 열 특성 분석 방식은 정확하지만, 설치 비용이 크고 보편적 적용이 어렵다는 한계가 있습니다.  
반면, 음향 센서는 **설치가 간편하고 비접촉 진단 가능**하므로 다양한 산업 장비에 적용하기 적합합니다.

하지만 음향 데이터는 **노이즈가 많고 비정형적**이므로, 정교한 특성 추출 및 머신러닝/딥러닝 기반 분류 알고리즘이 필요합니다.

**목표**는 음향 데이터를 이용한 이상 상태 분류 시스템을 개발하고, 실험적 타당성을 검증하는 것입니다.

### 1.3 프로젝트 목표

- 음향 데이터 수집 및 전처리
- 정상/이상 음향 특성 비교 및 특징 추출
- CNN 기반 분류 모델 개발
- 조기 이상 감지 기능 구현 및 산업 적용성 평가
- 스마트 팩토리 적용 가능성 제시

---

## 2. 주요 분석 단계 및 기법

| 단계 | 기법 | 목적 |
|------|------|------|
| 전처리 | MFCC, Mel-Spectrogram, 정규화 등 | 음향 데이터 정리 및 특징 추출 |
| 분류 | CNN, Random Forest, KNN 등 | 이상 상태 자동 분류 |
| 차원 축소 | PCA, t-SNE | 효율 향상 및 시각화 |
| 모델 선택 | Grid Search, Cross Validation | 성능 최적화 |
| 클러스터링 (선택) | k-Means, HDBSCAN | 이상 상태 정의 전 탐색용 |

> 📌 회귀(Regression)는 본 프로젝트에 부적합하므로 제외하였습니다. (분류 과제이므로)

---

## 3. 데이터 수집 및 전처리

### 3.1 데이터 수집

- 실제 장비의 정상/이상 음향 데이터 확보
- 이상 원인: 베어링 결함, 마찰 증가, 느슨함, 균열 등
- 다양한 작동 조건 반영 (속도, 하중, 온도 등)
- **공개 데이터셋 활용 가능**
  - ESC-50
  - MIMII Dataset (Hitachi)
  - CWRU Bearing Dataset
  - Amazon Lookout for Equipment

### 3.2 전처리

- **노이즈 제거**: Band-pass filter, Wavelet
- **정규화**: Min-Max, Z-score
- **특징 추출**:
  - Mel-Spectrogram
  - MFCC
  - Chroma, Spectral Centroid 등
- **데이터 증강**:
  - 잡음 추가, pitch shift, time stretch 등

---

## 4. 특성 추출 및 모델링

### 4.1 특징 추출 및 사용 데이터셋

- MIMII Dataset 중 fan 데이터를 활용
- **Mel-Spectrogram**: 2D 이미지 형태로 변환하여 모델에 입력
- Sampling rate: 16,000Hz, Mel-bins: 64로 설정

### 4.2 최종 분류 모델 (CRNN)

본 프로젝트에서는 CRNN(Convolutional Recurrent Neural Network)을 사용했습니다.

- CNN 층에서 주파수 패턴 특징을 추출
- LSTM 층을 통해 시간적 변화와 특징을 추가로 고려
- 최종 Dense 레이어를 통해 정상과 이상 상태를 분류

### 4.3 학습 과정

- Loss: CrossEntropyLoss
- Optimizer: Adam (learning rate=0.001)
- Epoch: 30회 수행  
  - Loss가 초기 약 98에서 최종적으로 약 3 수준으로 감소  
  - 과적합(Overfitting) 징후 없이 안정적인 수렴 확인

---

## 4.4 모델 성능 평가 (추가 테스트)

Kaggle의 [Anomaly Detection from Sound Data (Fan)](https://www.kaggle.com/datasets/vuppalaadithyasairam/anomaly-detection-from-sound-data-fan?resource=download)의 **Train Set**을 별도의 테스트 데이터로 사용하여 성능을 평가했습니다.

- 평가 데이터 총 개수: **6,521개**
- 평가 결과:
  - 정상: **4,889개 (75.0%)**
  - 이상: **1,632개 (25.0%)**

다음 이미지는 실제 평가 후 출력된 결과입니다:
![evaluation_result](evaluation_result.png)

---

## 4.5 학습 과정에서의 손실(loss) 감소 추이

다음 이미지는 실제 학습 과정 중 기록된 손실(loss) 값의 변화입니다:
![epoch_loss](epoch_loss.png)

---

## 4.6 정확도가 기대보다 낮은 이유 분석

- 학습 데이터 양 부족 및 불균형 문제
- Mel-Spectrogram 외 추가적 특징 미활용
- 모델 구조 및 복잡도 한계
- 하이퍼파라미터 최적화 부족

---

## 4.7 향후 개선 방향

- Transformer, Attention 기반 모델로의 확장 가능성 탐색
- 모델 경량화 및 모바일/엣지 디바이스 적용 가능성 고려
- 데이터 증강과 정교한 하이퍼파라미터 튜닝을 통한 성능 추가 향상

---

## 5. 참고자료

- Amazon Lookout for Equipment: [공식 블로그](https://aws.amazon.com/ko/blogs/korea/acoustic-anomaly-detection-using-amazon-lookout-for-equipment/)
- 공개 데이터셋: MIMII, CWRU 등
- Python 라이브러리: Librosa, Scipy, PyTorch

---

## 6. 결론

본 프로젝트에서는 CRNN 모델을 이용하여 기계 장비의 음향 데이터로 이상 상태를 분류하는 시스템을 구축하고 테스트를 진행했습니다. 비록 4.6에서 설명한 몇가지 요인 등으로 인해 정확도가 다소 하락하였지만 향후 추가적인 데이터 확보, 특징 추출의 다양화 및 고급 모델링 기법을 도입하여 정확도를 개선할 계획입니다.

---

## 🧑‍💻 역할 분담

- **멤버 1 임규원**: 코드 구현 및 데이터 처리
- **멤버 2 이재룡**: 자료 조사, 보고서 작성 및 동영상 녹화
