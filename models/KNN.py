from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def KNN(X_train, X_test, y_train, y_test):
    KN = KNeighborsClassifier()
    KN.fit(X_train, y_train)
    y_pred = KN.predict(X_test)

    # accuracy
    print("Logistic Regression Classifier")
    print("accuracy:", accuracy_score(y_pred, y_test)*100,"%")
    print()
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("classificaion report")
    print(classification_report(y_test, y_pred))