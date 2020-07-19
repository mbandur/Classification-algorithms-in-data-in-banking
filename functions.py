import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix as cm, classification_report

def vis_results(model, X_train, X_test, y_train, y_test):
        model.fit(X_train,y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)  
        y_probs = model.predict_proba(X_test)[:,1]
        
        report = classification_report(y_test, y_pred_test, output_dict = True)
        confusion_matrix = cm(y_test, y_pred_test)      
        AUC = roc_auc_score(y_test, y_probs) * 2 - 1
       
        print('Classification Report X_test:')
        print(classification_report(y_test, y_pred_test))
        print('Confusion Matrix:')
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 3)    
        sns.heatmap(confusion_matrix, annot = True, cmap = 'PuBu', linewidths = 2, square = True, fmt = 'n')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('Actual label')
        sns.despine()
    
        print('AUC: %.2f' % AUC)
        print(model)
        return model, report, AUC

def model_result(model_r):
    model_type = type(model_r[0]).__name__
    ACC = model_r[1]['accuracy']
    AUC = model_r[2]
    
    PRECISION_ALL = model_r[1]['weighted avg']['precision']
    RECALL_ALL = model_r[1]['weighted avg']['recall']
    F1_ALL = model_r[1]['weighted avg']['f1-score']
    
    PRECISION_1 = model_r[1]['1']['precision']
    RECALL_1 = model_r[1]['1']['recall']
    F1_1 = model_r[1]['1']['f1-score']
    
    return {model_type: [ACC, AUC, PRECISION_ALL, RECALL_ALL, F1_ALL, PRECISION_1, RECALL_1, F1_1]}   

def max_scores(data):
    max_scores = []
    for i in range(len(data.columns) - 1):
        col =  data[data.columns[i]]
        col_nam = data.columns[i]
        val = col[col.argmax()]
        ind = data.index[col.argmax()]
        max_scores.append([col_nam, ind, val])
    max_scores = pd.DataFrame(max_scores, columns=['score', 'model', 'value'])
    return max_scores