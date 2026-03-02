# Chapter 07. Ensemble Learning and Random Forests [Concept]

**태그:** #MachineLearning #Ensemble #Theory #BiasVariance

---

## 1. 개요 (Introduction)

여러 개의 예측기(Predictor)를 결합하여 개별 모델보다 더 높은 성능을 도출하는 기법. 이를 **'대중의 지혜(Wisdom of the Crowd)'**라 함. 보통 단일 모델의 성능이 한계에 도달했을 때 마지막에 성능을 쥐어짜기 위해 사용함.

---

## 2. 투표 기반 분류기 (Voting Classifiers)

### 2.1 하드 보팅 (Hard Voting)
- 각 모델이 예측한 클래스 중 다수결 투표로 최종 결정.
- **큰 수의 법칙 (Law of Large Numbers):** 51% 확률의 동전을 10,000번 던졌을 때 앞면이 다수일 확률이 97%가 넘는 것처럼, 약한 학습기(Weak Learner)도 충분히 많고 오차가 독립적이라면 강한 학습기(Strong Learner)로 수렴함.

### 2.2 소프트 보팅 (Soft Voting)
- 클래스 확률(`predict_proba()`)을 평균 내어 가장 높은 확률을 선택함.
- 신뢰도가 높은 투표에 가중치를 두는 효과가 있어 하드 보팅보다 일반적으로 성능이 우수. (SVC 사용 시 `probability=True` 설정 필수)

---

## 3. 배깅과 페이스팅 (Bagging and Pasting)

### 3.1 개념
- **Bagging (Bootstrap Aggregating):** 훈련 데이터에서 중복 허용하여 샘플링.
- **Pasting:** 중복 허용하지 않고 샘플링.

### 3.2 편향-분산 트레이드오프
- 개별 예측기는 전체 데이터가 아니므로 편향(Bias)이 높아질 수 있음.
- 하지만 앙상블 집계(Aggregation) 과정에서 오차가 상쇄되어 **분산(Variance)**이 크게 감소함.
- 일반적으로 배깅이 페이스팅보다 분산을 더 잘 줄여주어 널리 쓰임.

### 3.3 Out-of-Bag (OOB) 평가
- 배깅 샘플링 시 약 **37%**의 데이터는 선택되지 않음.
- 이 OOB 샘플을 별도의 검증 데이터셋처럼 사용하여 모델의 일반화 성능을 즉시 평가 가능.

---

## 4. 랜덤 포레스트 (Random Forests)

배깅 방식을 적용한 결정 트리의 앙상블 모델.

### 4.1 핵심: 특성 샘플링
- 노드 분할 시 모든 특성을 보는 게 아니라 무작위로 일부만 선택하여 최적 분할을 찾음.
- 트리 간 상관관계를 낮춰 분산을 더 줄임.

### 4.2 엑스트라 트리 (Extra-Trees)
- 분할 임계값(Threshold)까지 무작위로 설정.
- 최적 임계값 계산이 생략되어 학습 속도가 비약적으로 빠름.

---

## 5. 부스팅 (Boosting)

앞선 모델의 오차를 순차적으로 보정하여 강한 학습기를 구축.

### 5.1 에이다부스트 (AdaBoost)
- 잘못 분류된 샘플에 가중치를 부여하여 다음 모델이 이를 더 집중적으로 학습하게 함.

> [!formula] **AdaBoost 수식 핵심**
> 1. **에러율 ($r_j$):** $r_j = \frac{\sum_{i=1, \hat{y}_j^{(i)} 
eq y^{(i)}}^m w^{(i)}}{\sum_{i=1}^m w^{(i)}}$
> 2. **예측기 발언권 ($\alpha_j$):** $\alpha_j = \eta \log \frac{1-r_j}{r_j}$
> 3. **데이터 가중치 업데이트:** 틀린 샘플은 $w^{(i)} \exp(\alpha_j)$로 증폭하여 다음 학습에 전달.

### 5.2 그레이디언트 부스팅 (Gradient Boosting)
- 이전 모델의 **잔여 오차(Residual Error)**를 새로운 모델이 학습.
- **Shrinkage (축소):** 학습률(Learning Rate)을 낮게 잡아 성능을 올리되 많은 트리가 필요함. 조기 종료(Early Stopping)와 결합하여 최적의 트리 개수를 찾는 것이 관건.

---

## 6. 스태킹 (Stacking)

- 다수결 투표가 아닌, 예측기들의 결과를 입력값으로 사용하는 **메타 학습기(Meta Learner / Blender)**를 학습시킴.
- 복잡한 데이터 구조에서 앙상블의 시너지를 극대화할 때 유용함.
