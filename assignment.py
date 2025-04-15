from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time




X = df.drop(columns=['Borg'])
y = df['Borg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




knn = KNeighborsRegressor()

start_time_knn = time.time()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_time = time.time() - start_time_knn




dt = DecisionTreeRegressor()

start_time_dt = time.time()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_time = time.time() - start_time_dt




print("K-NN Results:")
print("MSE", mean_squared_error(y_test, y_pred_knn))
print("R²", r2_score(y_test, y_pred_knn))
print("Execution Time", knn_time)

print("\nDecision Tree Results:")
print("MSE", mean_squared_error(y_test, y_pred_dt))
print("R²", r2_score(y_test, y_pred_dt))
print("Execution Time", dt_time)



from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression




pipeline = Pipeline([        
    ('scaler', StandardScaler()),                           
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),
    ('regressor', DecisionTreeRegressor(random_state=42))   
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)




print("Improved Decision Tree Pipeline Results:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print("\nCross-Validation R² scores:", scores)
print("Mean CV R² score:", scores.mean())



from sklearn.ensemble import RandomForestRegressor



newPipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),
    ('regressor', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
])



newPipeline.fit(X_train, y_train)
y_pred = newPipeline.predict(X_test)



print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

cv_scores = cross_val_score(newPipeline, X, y, cv=3, scoring='r2')
print("CV R² Scores:", cv_scores)
print("Mean CV R² Score:", cv_scores.mean())


