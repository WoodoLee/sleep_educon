from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def GradientBoosting(X_train, X_test, y_train, y_test):
    gb = GradientBoostingClassifier(criterion='entropy', max_depth=32, bootstrap=True, random_state=42, n_estimators=256, min_samples_split=2)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    # accuracy
    print("Gradient Boosting Classifier")
    print("accuracy:", accuracy_score(y_pred, y_test)*100,"%")
    print()
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("classificaion report")
    print(classification_report(y_test, y_pred))
