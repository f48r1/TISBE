from sklearn import metrics as _metrics
import pandas as pd

def PRcurve(y, scores):
    precision, recall, thresholds = _metrics.precision_recall_curve(y, scores, pos_label=1)
    return _metrics.auc(recall, precision)

def ROCcurve(y, scores):
    fpr, tpr, thresholds = _metrics.roc_curve(y, scores, pos_label=1)
    return _metrics.auc(fpr, tpr)

# x = true Y
# y = predicted Y
# z = classification score
metrics = dict(
    spec = lambda x,y,z : _metrics.recall_score(x,y, pos_label=0),
    sens = lambda x,y,z : _metrics.recall_score(x,y, pos_label=1),
    balacc = lambda x,y,z: _metrics.balanced_accuracy_score(x,y),
    mcc= lambda x,y,z: _metrics.matthews_corrcoef(x,y),
    ppv = lambda x,y,z : _metrics.precision_score(x,y, pos_label=1),
    npv = lambda x,y,z :_metrics.precision_score(x,y, pos_label=0),
    PRcurve = lambda x,y,z : PRcurve(x,z),
    ROCcurve = lambda x,y,z: ROCcurve(x,z),
    # you can add more metrics in case...
              )

metricsAll = {
    "tp" : lambda x,y,z: sum((x==y) & (x==1)),
    "tn" : lambda x,y,z: sum((x==y) & (x==0)),
    "fp" : lambda x,y,z: sum((x!=y) & (x==0)),
    "fn" : lambda x,y,z: sum((x!=y) & (x==1)),
    **metrics,
}

def getTableFromStats(stats):
    dfOut = pd.DataFrame(index=stats.columns)
    for metric,(median,first,third) in zip(stats.columns,
                                           stats.loc[["50%","25%","75%"],:].values.T):
        dfOut.loc[metric,"median"]=f"{median}"
        dfOut.loc[metric,"(1st quartile, 3rd quartile)"]=f"({first}, {third})"
    return dfOut