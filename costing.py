import fairness as fr
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import sklearn.model_selection
import pandas as pd
import numpy as np


class CostingFairness(fr.Fairness):

    def __init__(self,
                 input_dataframe:pd.DataFrame,
                 label_names,
                 protected_attribute_names,
                 trained_model: sklearn.linear_model = None,
                 predictions: pd.DataFrame = None):

        super().__init__(input_dataframe=input_dataframe,
                         label_names=label_names,
                         protected_attribute_names=protected_attribute_names,
                         privileged_groups=None,
                         unprivileged_groups=None)

        if trained_model is None:
            X = self.data.loc[:, self.data.columns != self.label_names[0]]
            y = self.data.loc[:, self.label_names]
            self.model = sklearn.linear_model.LogisticRegression().fit(X, y)
        else:
            self.model = trained_model

        if predictions is None:
            self.y_pred = self.predict()
            self.y_proba = self.predict_proba()
        else:
            self.y_proba = predictions


    def pred_on_threshold(self, threshold, y_proba):
        preds = [1 if (i > threshold) else 0 for i in y_proba[:,1]]

        return preds

    def predict(self):
        X = self.data.loc[:, self.data.columns != self.label_names[0]]
        try:
            y_pred = self.model.predict(X=X)
        except TypeError:
            y_pred = self.model.predict(X=X.values, s=self.data[self.protected_attribute_names].values)

        return y_pred

    def predict_proba(self):
        X = self.data.loc[:, self.data.columns != self.label_names[0]]
        try:
            y_proba = self.model.predict_proba(X=X)
        except TypeError:
            y_proba = self.model.predict_proba(X=X.values, s=self.data[self.protected_attribute_names].values)

        return y_proba

    def cal_fair(self, preds=None):

        data = self.data.copy()

        y_pred = 'preds'
        # print(y_pred)

        if preds is None:
            data[y_pred] = self.y_pred
        else:
            data[y_pred] = preds

        #print(self.y_proba)


        priv = data[data[self.protected_attribute_names[0]] == 1]
        unpriv = data[data[self.protected_attribute_names[0]] == 0]
        acc_priv = sklearn.metrics.accuracy_score(y_true=priv[self.label_names], y_pred=priv[y_pred])

        acc_unpriv = sklearn.metrics.accuracy_score(y_true=unpriv[self.label_names], y_pred=unpriv[y_pred])
        acc_diff = abs(acc_priv - acc_unpriv)

        auc_priv = sklearn.metrics.roc_auc_score(y_true=priv[self.label_names], y_score=priv[y_pred])
        auc_unpriv = sklearn.metrics.roc_auc_score(y_true=unpriv[self.label_names], y_score=unpriv[y_pred])





        auc_diff = abs(auc_priv - auc_unpriv)

        f1_score_priv = sklearn.metrics.f1_score(y_true=priv[self.label_names], y_pred=priv[y_pred])
        f1_score_unpriv = sklearn.metrics.f1_score(y_true=unpriv[self.label_names], y_pred=unpriv[y_pred])
        f1_score_diff = abs(f1_score_priv - f1_score_unpriv)
        precision_score_priv = sklearn.metrics.precision_score(y_true=priv[self.label_names], y_pred=priv[y_pred])
        precision_score_unpriv = sklearn.metrics.precision_score(y_true=unpriv[self.label_names], y_pred=unpriv[y_pred])
        precision_score_diff = abs(precision_score_priv - precision_score_unpriv)



        return acc_diff, auc_diff, f1_score_diff, precision_score_diff

    def cal_cost_acc_fair_auc_f1_precision_per_threshold(self, threshold, cost_per_fp, cost_per_fn):

        preds = self.pred_on_threshold(threshold, self.y_proba)

        tn, fp, fn, tp = confusion_matrix(self.data[self.label_names], preds).ravel()
        cost = cost_per_fp * fp + cost_per_fn * fn

        accuracy = sklearn.metrics.accuracy_score(y_true=self.data[self.label_names], y_pred=preds)
        acc_diff, auc_diff, f1_score_diff, precision_score_diff = self.cal_fair(preds=preds)



        auc = sklearn.metrics.roc_auc_score(y_true=self.data[self.label_names], y_score=preds) #######TRUE N SCORES##




        f1_score = sklearn.metrics.f1_score(y_true=self.data[self.label_names], y_pred=preds) ########## BOTH BINARY ####




        precision_score = sklearn.metrics.precision_score(y_true=self.data[self.label_names], y_pred=preds) ######  BOTH BINARY######
        # #print(threshold, precision_score)
        #




        return cost, accuracy, auc, f1_score, precision_score, acc_diff, auc_diff, f1_score_diff, precision_score_diff

    def calculate_cost_matrix(self, false_positive_cost, false_negative_cost):
        thresholds = np.linspace(0.3, 0.8, 50)
        cost_acc_fair_per_threshold = [
            self.cal_cost_acc_fair_auc_f1_precision_per_threshold(threshold=threshold, cost_per_fp=false_positive_cost,
                                                                  cost_per_fn=false_negative_cost) for
            threshold in thresholds]

        cost, acc, auc, f1_score, precision_score, acc_diff, auc_diff, f1_score_diff, precision_score_diff = zip(
            *cost_acc_fair_per_threshold)

        fdf = pd.DataFrame(
            {'thresh': thresholds,
             'cost': cost,
             'accuracy': acc,
             'auc': auc,
             'f1': f1_score,
             'prec': precision_score,
             'acc_diff': acc_diff,
             'auc_diff': auc_diff,
             'f1_diff': f1_score_diff,
             'prec_diff': precision_score_diff
             })

        return fdf

    def return_cost_fairness_accuracy_optimised(self, false_negative_cost=300, false_positive_cost=700):

        cost_matrix = self.calculate_cost_matrix(false_negative_cost=false_negative_cost, false_positive_cost=false_positive_cost)
        output_df = pd.DataFrame(columns=list(cost_matrix.columns.values).append('OPT'))

        cost_opt = cost_matrix[cost_matrix.cost == cost_matrix.cost.min()]
        cost_opt['OPT'] = 'COST'
        output_df = output_df.append(cost_opt)

        acc_opt = cost_matrix[cost_matrix.accuracy == cost_matrix.accuracy.max()]
        acc_opt['OPT'] = 'ACC'
        output_df = output_df.append(acc_opt)

        fair_opt = cost_matrix[cost_matrix.auc_diff == cost_matrix.auc_diff.min()].reset_index(drop=True).loc[0,:]
        fair_opt['OPT'] = 'FAIR'
        output_df = output_df.append(fair_opt)

        # output_df['OPT'] = ['COST', 'ACC', 'FAIR']
        output_df.reset_index(inplace=True)
        del output_df['index']
        return output_df
