# Chapter 07. Ensemble Learning and Random Forests [Experiment]

**태그:** #Experiment #Analysis #SystemOptimization #Benchmarking

---

## 1. 실험 설계 (Experiment Design)

앙상블 모델은 단일 모델보다 성능이 우수하지만, 모델 개수($n\_estimators$)가 늘어남에 따라 학습 및 추론 시간이 선형적으로 증가함. 시스템 자원이 한정된 환경에서의 최적점을 찾는 실험 설계.

### 1.1 가설 설정
1. $n\_estimators$가 특정 임계값을 넘으면 정확도 향상폭은 둔화되지만 연산 비용은 계속 증가할 것이다.
2. 소프트 보팅이 하드 보팅보다 높은 정확도를 유지하면서도 더 적은 모델 개수로 동일한 성능을 낼 것이다.
3. 엑스트라 트리는 랜덤 포레스트 대비 학습 시간을 최소 30% 이상 단축시킬 것이다.

---

## 2. 스트레스 테스트 및 벤치마킹 (Benchmarking)

### 2.1 모델 규모에 따른 자원 사용량 분석
- **대상:** RandomForest, XGBoost, LightGBM
- **지표:** CPU Usage, Memory Footprint, Inference Latency (ms)
- **실험 데이터:** Moons 데이터셋 (또는 고차원 특성 데이터)

### 2.2 OOB 평가 vs 교차 검증 (Cross-Validation)
- 배깅에서 OOB 점수가 실제 교차 검증 점수와 얼마나 일치하는지 비교.
- OOB 평가를 통해 검증 시간을 얼마나 단축할 수 있는지 측정.

---

## 3. 결과 분석 및 한계 (Analysis)

- **오버피팅 관점:** 부스팅 계열 모델에서 학습률을 극단적으로 낮추었을 때 조기 종료 시점이 어떻게 변하는지 분석.
- **병렬화 효율:** `n_jobs=-1` 설정 시 코어 개수에 따른 배깅의 속도 향상(Speedup) 비율 측정. 아카이브 시스템에서는 멀티 코어 활용도가 랜덤 포레스트의 가장 큰 장점 중 하나임.

---

## 4. 향후 과제 (Future Works)

- **XGBoost Hyperparameter Tuning:** `max_depth`와 `learning_rate`의 상관관계 분석을 통한 최적화.
- **Stacking 성능 실험:** 서로 다른 유형의 모델(SVM, RF, KNN)을 결합했을 때의 상호 보완 효과 실증 연구.
