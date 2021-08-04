# Thanks to "Dennis Trimarchi"  from https://github.com/DTrimarchi10/confusion_matrix
import numpy                       as np
import matplotlib.pyplot           as plt
import seaborn                     as sns
import pandas                      as pd
from sklearn.metrics               import auc, precision_recall_curve
from sklearn.metrics               import average_precision_score
from sklearn.model_selection       import RepeatedStratifiedKFold
from sklearn.metrics               import make_scorer
from sklearn.model_selection       import GridSearchCV
from sklearn.metrics               import fbeta_score
from sklearn.metrics               import recall_score




import matplotlib.pyplot           as plt

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''

    font = {'weight' : 'bold',
        'size'   : 14}

    plt.rc('font', **font)
    plt.rcParams['figure.figsize'] = (10.0, 8.0)

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])  # TP/(TP+FN) = sensitivity
            specificity= cf[0,0] / sum(cf[0,:]) # TN/(TN+FP)
            balanced_accuracy = (recall+specificity)/2
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nBalanced_accuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,balanced_accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
        
def groupby_(dataFrame, col, col_name):
    p=dataFrame.groupby(col).size()
    s=dataFrame[col].value_counts(normalize=True,sort=False).mul(100) 
    display(pd.DataFrame({'#'+col_name:s.index, '+':p.values, '%':s.values }))
    



#=================================
#        model comparision
#=================================
def PRC_model(X_tr, y_tr, X_tst, y_tst,  model):
    # define evaluation procedure
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_tst)[:, 1]    

    precision, recall, _ = precision_recall_curve(y_tst, probs)
    auc_ = auc(recall, precision)
    
    prediction = model.predict(X_tst)
    average_precision = average_precision_score(y_tst, prediction)

    return precision, recall, auc_, average_precision

def PRC_models(models, names, results,X_train, y_train, X_test, y_test):
    baseline_model = sum(y_test == 1) / len(y_test)
    
    plt.figure(figsize=(20, 10))
    plt.plot([0, 1], [baseline_model, baseline_model], linestyle='--', label='Baseline model')    
    
    # evaluate each model
    for i in range(len(models)):
        # evaluate the model and store results
        precision, recall, auc_, average_precision = PRC_model(X_train, y_train, X_test, y_test, models[i])
        plt.plot(recall, precision, label='AUC ({}): {:.2f}'.format(names[i], auc_))
        print('{}: precision: {}, recall: {}, auc_: {}, Average precision-recall: {}'.format(names[i],np.mean(precision),np.mean(recall), auc_, average_precision))
        
     
    # plot the results
    plt.title('Precision-Recall Curve', size=20)
    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    plt.legend();
    plt.show()
    
#=================================
#        evaluate a model
#=================================
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    # define the model evaluation metric
    metric = make_scorer(recall_score)#sensitivity_score)
    #metric = make_scorer(fbeta_score, beta=2)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
    return scores

def evaluate_models(models, names, results, X, y):
    # evaluate each model
    for i in range(len(models)):
        # evaluate the model and store results
        scores = evaluate_model(X, y, models[i])
        results.append(scores)
        # summarize and store
        print('>%s %.3f (%.3f)' % (names[i], np.mean(scores), np.std(scores)))
    # plot the results
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()
    
#=================================
#        gridsearch
#=================================
def gc_model(X_tr , y_tr, X_tst, y_tst, model, space):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    # define the model evaluation metric
    metric = make_scorer(recall_score)#sensitivity_score)
    #metric = make_scorer(fbeta_score, beta=2)
    # evaluate model
    search = GridSearchCV(estimator=model, param_grid=space, n_jobs=-1, cv=cv, scoring=metric,
                          error_score=0,verbose=5)
    search_result = search.fit(X_tr , y_tr)
    print('Training set score: ' + str(search.score(X_tr , y_tr)))
    print('Test set score: ' + str(search.score(X_tst, y_tst)))
    return search_result

def gc_models(models, names, results, spaces,X_tr , y_tr, X_tst, y_tst):
        
    # evaluate each model
    for i in range(len(models)):
        # evaluate the model and store results
        search_result = gc_model(X_tr , y_tr, X_tst, y_tst, models[i], spaces[i])
        #results.append(scores)
        # summarize and store
        print("Best of %s: %f using %s \n" % (names[i], search_result.best_score_, search_result.best_params_))