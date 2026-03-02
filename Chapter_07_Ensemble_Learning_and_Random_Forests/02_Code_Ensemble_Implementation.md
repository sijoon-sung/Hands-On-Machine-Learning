# Chapter 07. Ensemble Learning and Random Forests [Code]

**태그:** #ScikitLearn #Implementation #Python #Modeling

---

## 1. 투표 기반 분류기 (Voting Classifier)

여러 분류기(Logistic, RF, SVC 등)를 결합하여 성능을 측정하는 예시입니다.

### 1.1 개별 모델 및 앙상블 정의
```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True) # Soft Voting을 위해 활성화

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft' # 확률 기반 투표
)
```

### 1.2 모델 학습 및 평가 결과 분석
각 모델과 앙상블의 정확도를 비교하면 앙상블의 성능이 가장 높음을 알 수 있습니다.

| 모델명 | 정확도(Accuracy) |
| :--- | :--- |
| LogisticRegression | 0.864 |
| RandomForestClassifier | 0.896 |
| SVC | 0.888 |
| **VotingClassifier** | **0.904** |

---

## 2. 배깅과 OOB 평가 (Bagging & OOB)

### 2.1 배깅 분류기 구현
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators=500,        # 트리 500개
    max_samples=100,         # 각 트리당 100개 샘플
    bootstrap=True,          # 배깅(복원 추출) 활성화
    n_jobs=-1,               # 병렬 처리
    oob_score=True           # OOB 평가 활성화
)
bag_clf.fit(X_train, y_train)
```

### 2.2 결과 해석
- **OOB 점수:** 약 **90.1%**로, 별도의 검증 세트 없이 모델 성능을 잘 추정함.
- **테스트 세트 정확도:** 약 **91.2%**로 OOB 점수와 유사하게 도출됨.
- 단일 트리보다 결정 경계가 매끄럽고 일반화 성능이 우수함.

---

## 3. 랜덤 포레스트와 특성 중요도

### 3.1 랜덤 포레스트 학습
```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
```

### 3.2 Iris 데이터셋 특성 중요도 분석
학습 후 어떤 특징이 예측에 결정적이었는지 수치로 확인 가능합니다.
- **Petal Length:** 약 44% (가장 중요)
- **Petal Width:** 약 42%
- **Sepal Length:** 약 11%
- **Sepal Width:** 약 2%

---

## 4. 부스팅 (Boosting)

### 4.1 에이다부스트 (AdaBoost)
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), # 깊이 1인 트리(Stump) 사용
    n_estimators=200, 
    algorithm="SAMME.R", 
    learning_rate=0.5
)
ada_clf.fit(X_train, y_train)
```

### 4.2 그레이디언트 부스팅 (Gradient Boosting)
```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 120개 트리로 초기 학습
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

# 조기 종료를 위한 최적의 트리 개수 계산
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1
```
