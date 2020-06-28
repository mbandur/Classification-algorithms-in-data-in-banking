import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, confusion_matrix as cm, classification_report, mean_absolute_error

def upsample(X, y):
    df_all = pd.concat((X, pd.DataFrame({'y': y}, index = y.index)), axis = 1)
    
    df_majority = df_all [df_all.y == 0]
    df_minority = df_all[df_all.y == 1]
    df_minority_upsampled = resample(df_minority, replace = True, n_samples = df_majority.shape[0], random_state = 100)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled], axis=0)
    y_upsampled = df_upsampled.y
    X_upsampled = df_upsampled.drop('y', axis=1)

    return X_upsampled, y_upsampled
    
def vis_results(model, X_train, X_test, y_train, y_test):
        model.fit(X_train,y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)
        
        report = classification_report(y_test, y_pred_test, output_dict = True)
        confusion_matrix = cm(y_test, y_pred_test)      
        AUC = roc_auc_score(y_test, y_pred_test) * 2 - 1
      
        mae = mean_absolute_error(y_pred_test, y_test)
 
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
        print('MAE: %.2f' % mae)

        print(model)
        return model, report, AUC, mae

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
    
    MAE = model_r[3]
    return {model_type: [ACC, AUC, PRECISION_ALL, RECALL_ALL, F1_ALL, PRECISION_1, RECALL_1, F1_1, MAE]}   
