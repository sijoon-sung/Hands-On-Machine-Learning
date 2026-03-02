# Chapter 07. Ensemble Learning and Random Forests [Code]

**태그:** #ScikitLearn #Implementation #Python #Modeling

---

## 1. 투표 기반 분류기 (Voting Classifier)

여러 모델을 결합하여 예측 수행. 소프트 보팅이 가능한 모델은 `probability=True` 설정이 필요할 수 있음.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True) # 소프트 보팅을 위해 확률 예측 활성화

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
```

---

## 2. 배깅과 OOB 평가 (Bagging & OOB)

`bootstrap=True`를 통해 배깅(복원 추출)을 활성화하고, `oob_score=True`로 자동 검증 수행.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True
)
bag_clf.fit(X_train, y_train)

# OOB 평가 점수 확인
print(f"OOB Score: {bag_clf.oob_score_}")
```

---

## 3. 랜덤 포레스트와 특성 중요도 (Random Forest & Importance)

랜덤 포레스트는 학습 후 특성별 중요도를 자동으로 산출함.

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])

# 특성 중요도 출력
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
```

---

## 4. 에이다부스트 (AdaBoost)

깊이가 1인 결정 트리(Decision Stump)를 기본 학습기로 사용하는 경우가 많음.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5
)
ada_clf.fit(X_train, y_train)
```

---

## 5. 그레이디언트 부스팅과 조기 종료 (GBRT & Early Stopping)

검증 오차가 향상되지 않을 때 학습을 멈추어 과적합 방지.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

# 오차가 가장 적은 트리 개수 찾기
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

# 최적의 트리 개수로 다시 모델 생성
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
```
