from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
def build_model(train_data, model_object):
    X_train, y_train = train_data
   
    model_object.fit(X_train, y_train)
    return model_object

def evaluate_model(train_data, test_data, trained_model,roc_analysis=True):
    X_train, y_train = train_data
    X_test, y_test = test_data


    # Get Predictions
    y_train_pred = trained_model.predict(X_train)
    y_test_pred = trained_model.predict(X_test)

    # Get Predicted Probabilities (for ROC/AUC)
    y_train_prob = trained_model.predict_proba(X_train)[:, 1]
    y_test_prob = trained_model.predict_proba(X_test)[:, 1]

    # Print Classification Reports
    print("Train Classification Report:")
    print(classification_report(y_train, y_train_pred))
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    if roc_analysis==True:
        # Plot ROC Curve for both Train and Test
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_prob)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    
        auc_train = auc(fpr_train, tpr_train)
        auc_test = auc(fpr_test, tpr_test)
        
        roc_auc = auc(fpr_train, tpr_train)
        # Compute Youden's J statistic for each threshold
        youden_j = tpr_train - fpr_train
        optimal_threshold_index = np.argmax(youden_j)
        optimal_threshold = thresholds_train[optimal_threshold_index]
        
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        plt.figure(figsize=(10,5))
        plt.plot(fpr_train, tpr_train, label=f'Train ROC (AUC = {auc_train:.2f})')
        plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {auc_test:.2f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    return None
