# Machine Learning to Systems (MLSys) Research Log

이 레포지토리는 『Hands-On Machine Learning (2nd Edition)』을 기반으로 한 머신러닝 및 딥러닝 학습 기록이자, 이를 시스템 엔지니어링 및 클라우드 환경 최적화 관점에서 분석한 연구 로그입니다. 

단순한 알고리즘 구현을 넘어 모델의 수학적 원리 파악, 메모리 병목 분석, 그리고 대규모 데이터 처리 시의 한계를 극복하는 실험 과정에 집중합니다.

## Architecture & Workflow

각 챕터는 시스템 설계와 분석의 기본이 되는 3단 구조로 아카이빙됩니다.

1. [Concept]: 알고리즘의 수학적 증명, 비용 함수, 시간/공간 복잡도(Big-O) 분석
2. [Code]: Scikit-Learn, TensorFlow를 활용한 베이스라인 파이프라인 구축
3. [Experiment]: 알고리즘의 한계 노출(Stress Test), 하이퍼파라미터 튜닝, 메모리/추론 속도(Latency) 최적화 등 논문 및 시스템 아키텍처 관점의 한계 극복 실험

---

## Map of Content (MOC)

### Part 1: The Fundamentals of Machine Learning

* Chapter 01. The Machine Learning Landscape (머신러닝의 풍경)
  * [Concept] 머신러닝 시스템의 분류 및 주요 도전 과제
  * [Experiment] 모델 복잡도에 따른 과대적합 발생 및 데이터 양의 비합리적 효과성 증명

* Chapter 02. End-to-End ML Project (머신러닝 프로젝트 처음부터 끝까지)
  * [Code] 데이터 적재, 결측치 처리, 특성 스케일링을 아우르는 전체 파이프라인 구축

* Chapter 03. Classification (분류)
  * [Concept] 오차 행렬, 정밀도/재현율 트레이드오프, ROC 곡선

* Chapter 04. Training Models (모델 훈련)
  * [Concept] 경사 하강법(BGD, SGD, Mini-batch)의 수학적 이해 및 규제 선형 모델

* Chapter 05. Support Vector Machines (서포트 벡터 머신)
  * [Experiment] 선형 SVM과 커널 트릭의 분류 시간 복잡도(O(m^3)) 한계 분석

* Chapter 06. Decision Trees (결정 트리)
  * [Concept] CART 알고리즘 원리, 불순도(Gini/Entropy) 지표 분석
  * [Experiment] 트리의 훈련 병목 및 실시간 네트워크 패킷 추론 속도 실험

* Chapter 07. Ensemble Learning and Random Forests (앙상블 학습과 랜덤 포레스트)
  * [[Chapter_07_Ensemble_Learning_and_Random_Forests/07_Ensemble_Learning_Concepts|[Concept] 배깅, 페이스팅, 부스팅(AdaBoost, Gradient Boosting)]]

* Chapter 08. Dimensionality Reduction (차원 축소)
  * [Code] PCA, t-SNE 알고리즘을 통한 고차원 데이터의 저차원 투영 및 시각화

* Chapter 09. Unsupervised Learning (비지도 학습)
  * [Concept] 군집화(K-Means, DBSCAN) 및 가우시안 혼합 모델

---

### Part 2: Neural Networks and Deep Learning
> Note: 향후 학습 진행에 따라 지속적으로 업데이트될 섹션입니다.

* Chapter 10~11. Keras/TensorFlow 심층 신경망 훈련
  * [Concept] 그래디언트 소실/폭주 문제와 활성화 함수, 배치 정규화

* Chapter 13. TensorFlow Data API (데이터 전처리 및 적재)
  * [Experiment] tf.data API를 활용한 GPU 병목 해소 및 I/O 파이프라인 최적화 연구

* Chapter 19. Training & Deploying at Scale (대규모 모델 훈련 및 배포)
  * [Experiment] 분산 학습 아키텍처 및 FastAPI를 활용한 실시간 모델 서빙 API 구축

---

## Tech Stack & Environment

* Languages: Python (Pandas, NumPy)
* Frameworks: Scikit-Learn, TensorFlow/Keras, FastAPI
* Knowledge Management: Obsidian, GitHub Projects