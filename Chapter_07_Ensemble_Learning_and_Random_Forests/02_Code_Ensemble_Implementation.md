# Chapter 07. Ensemble Learning and Random Forests [Code]

**태그:** #ScikitLearn #Implementation #Python #Modeling

---

## 1. 투표 기반 분류기 (Voting Classifier)

여러 분류기를 결합하여 투표를 통해 최종 예측을 수행하는 코드와 상세 분석입니다.

### 1.1 라이브러리 임포트 및 객체 생성
```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. 개별 분류기 객체 생성
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True) # Soft Voting을 위해 활성화

# 2. 투표 분류기 정의
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)
```

### 1.2 상세 요소 설명
- **`log_clf`, `rnd_clf`, `svm_clf`:** 각각 로지스틱 회귀, 랜덤 포레스트, 서포트 벡터 머신 모델을 생성합니다.
- **`estimators`:** 앙상블에 참여할 모델들의 리스트입니다 (이름, 객체 쌍).
- **`voting='soft'`:** 각 모델의 클래스 확률을 평균 내어 다수결을 정합니다. (`hard`는 단순 결과 다수결)

### 1.3 모델 학습 및 평가 결과 분석
```python
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

**[실행 결과 분석]**
| 모델명 | 정확도(Accuracy) |
| :--- | :--- |
| LogisticRegression | 0.864 |
| RandomForestClassifier | 0.896 |
| SVC | 0.888 |
| **VotingClassifier** | **0.904** |

- 단일 모델 중 가장 성능이 좋은 것은 랜덤 포레스트(89.6%)이나, 세 모델을 합친 **VotingClassifier가 90.4%로 가장 높은 성능**을 보입니다.
- **결론:** 약한 학습기들을 결합하면 더 강력한 학습기가 될 수 있다는 효과를 입증합니다.

---

## 2. 배깅과 페이스팅 (Bagging & Pasting)

훈련 데이터의 무작위 부분집합을 다르게 구성하여 학습시키는 방식입니다.

### 2.1 Bagging 구현 예시
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier  

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),   # 기본 분류기: 의사결정 트리
    n_estimators=500,           # 트리 개수 (500개)
    max_samples=100,            # 각 트리가 학습할 샘플 수 (100개)
    bootstrap=True,             # 샘플링 방식: True → 배깅, False → 페이스팅
    n_jobs=-1                   # CPU 병렬 처리: 모든 코어 사용
)
bag_clf.fit(X_train, y_train)
```

### 2.2 코드 요소 설명 및 결과 해석
- **`DecisionTreeClassifier()`:** 기본 학습기를 의사결정 트리로 설정.
- **`n_estimators=500`:** 총 500개의 트리를 학습시켜 안정적인 결과를 얻음.
- **`max_samples=100`:** 훈련 데이터에서 무작위로 100개를 뽑아 학습하여 다양성 확보.
- **`bootstrap=True`:** 샘플링을 **중복 허용**하여 수행(배깅).
- **`n_jobs=-1`:** 모든 CPU 코어를 사용해 대규모 데이터에서도 빠른 학습 가능.

**[결과 해석]**
- BaggingClassifier는 기본적으로 **소프트 보팅(Soft Voting)**을 수행합니다.
- 단일 트리보다 앙상블의 결정 경계가 훨씬 **매끄럽고 일반화 성능이 좋음**. 분산이 줄어들어 과적합 위험이 낮아집니다.
![[Pasted image 20260202183030.png]]

---

## 3. OOB (Out-of-Bag) 평가

별도의 검증 세트 없이 배깅의 선택되지 않은 데이터를 활용한 평가입니다.

### 3.1 OOB 평가 코드
```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    bootstrap=True,
    n_jobs=-1,
    oob_score=True        # OOB 평가 활성화
)
bag_clf.fit(X_train, y_train)
print(f"OOB Score: {bag_clf.oob_score_}")
```

### 3.2 추가 기능
- `bag_clf.oob_decision_function_`: 각 훈련 인스턴스에 대해 OOB 평가 시 추정된 **클래스 확률**을 반환합니다.

---

## 4. 랜덤 포레스트 (Random Forests)

결정 트리의 앙상블로, 노드 분할 시 무작위성을 추가한 모델입니다.

### 4.1 기본 구현
```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
```

### 4.2 하이퍼파라미터 상세 구분
- **1. DecisionTreeClassifier의 파라미터 (트리 성장 제어):**
    - `max_depth`, `max_leaf_nodes`, `min_samples_split`, `criterion` 등.
- **2. BaggingClassifier의 파라미터 (앙상블 제어):**
    - `n_estimators`, `bootstrap`, `max_samples`, `max_features`, `n_jobs` 등.

### 4.3 BaggingClassifier로 구현한 유사 모델
```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
)
```
- **`splitter="random"`:** 무작위로 선택된 특징 중에서 분할을 결정하여 Random Forest와 거의 동일하게 동작하게 함.

---

## 5. 특성 중요도 (Feature Importance)

어떤 특징이 예측에 결정적인 역할을 했는지 분석하는 코드입니다.

### 5.1 Iris 데이터셋 예시
```python
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
```

**[출력 결과 및 활용]**
- sepal length: ~11%, sepal width: ~2%, petal length: ~44%, petal width: ~42%
- **분석:** 꽃잎(petal) 관련 특징이 꽃받침(sepal)보다 훨씬 중요함을 알 수 있습니다.
- **활용:** MNIST 이미지 데이터에 적용하면 픽셀 중요도를 시각화할 수 있으며, **특징 선택(Feature Selection)**에 매우 유용합니다.
![[Pasted image 20260214144459.png]]

---

## 6. 부스팅 (Boosting)

### 6.1 AdaBoost 구현
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), # Decision Stump
    n_estimators=200, algorithm="SAMME.R", learning_rate=0.5
)
ada_clf.fit(X_train, y_train)
```

### 6.2 Gradient Boosting 조기 종료 (Early Stopping)
```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

# staged_predict를 사용하여 각 트리 단계별 오차 측정
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1
```
