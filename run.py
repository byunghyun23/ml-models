from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


### Load data
raw_wine = datasets.load_wine()

X = raw_wine.data
y = raw_wine.target

X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)


### Scaling
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)


### models
dict_acc = {}

# 1. RidgeClassifier
ridge = RidgeClassifier()
ridge.fit(X_tn_std, y_tn)
pred_ridge = ridge.predict(X_te_std)
acc_ridge = accuracy_score(y_te, pred_ridge)
dict_acc['RidgeClassifier'] = acc_ridge

# 2. GaussianNB
gnb = GaussianNB()
gnb.fit(X_tn_std, y_tn)
pred_gnb = gnb.predict(X_te_std)
acc_gnb = accuracy_score(y_te, pred_gnb)
dict_acc['GaussianNB'] = acc_gnb

# 3. DecisionTreeClassifier
dct = DecisionTreeClassifier(random_state=0)
dct.fit(X_tn_std, y_tn)
pred_dct = dct.predict(X_te_std)
acc_dct = accuracy_score(y_te, pred_dct)
dict_acc['DecisionTreeClassifier'] = acc_dct

# 4. SVC
svc = SVC(kernel='linear', random_state=0)
svc.fit(X_tn_std, y_tn)
pred_svc = svc.predict(X_te_std)
acc_svc = accuracy_score(y_te, pred_svc)
dict_acc['SVC'] = acc_svc

# 5. XGBClassifier
xgb = XGBClassifier(gamma=0.5, learning_rate=0.03, max_depth=32, n_estimators=1000)
xgb.fit(X_tn_std, y_tn)
pred_xgb = xgb.predict(X_te_std)
acc_xgb = accuracy_score(y_te, pred_xgb)
dict_acc['XGBClassifier'] = acc_xgb

# 6. GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=1800,
                                    learning_rate=0.01636280613755809,
                                    max_depth=32,
                                    max_features='sqrt',
                                    min_samples_leaf=5,
                                    min_samples_split=9,
                                    random_state=1,
                                    validation_fraction=0.3,
                                    n_iter_no_change=100)
gb.fit(X_tn_std, y_tn)
pred_gb = gb.predict(X_te_std)
acc_gb = accuracy_score(y_te, pred_gb)
dict_acc['GradientBoostingClassifier'] = acc_gb

# 7. RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_tn_std, y_tn)
pred_rfc = rfc.predict(X_te_std)
acc_rfc = accuracy_score(y_te, pred_rfc)
dict_acc['RandomForestClassifier'] = acc_rfc

# 8. LGBMClassifier
lgb = LGBMClassifier()
lgb.fit(X_tn_std, y_tn)
pred_lgb = lgb.predict(X_te_std)
acc_lgb = accuracy_score(y_te, pred_lgb)
dict_acc['LGBMClassifier'] = acc_lgb


### results
for model, acc in sorted(dict_acc.items()):
    print(model, acc)
