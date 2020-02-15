import pandas as pd
import numpy as np

import warnings
import sys
from fpdf import FPDF

import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model

pd.set_option('display.max_columns', None)
sys.path.append("../")
sys.path.insert(1, "../")
warnings.filterwarnings('ignore')
np.random.seed(6969)
#
# C_TP = 0
# C_FP = 700
# C_TN = 0
# C_FN = 300


class PDF(FPDF):
    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Select Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Print centered page number
        self.cell(0, 0, 'Page %s' % self.page_no(), 0, 0, 'C')

    def write_table_to_pdf(self, dataframe: pd.DataFrame):

        rows, cols = dataframe.shape

        if rows > 20:
            raise OverMaxRowsException("Can't convert more that 20 rows on to pdf")

        cell_width = (self.fw-20)/cols
        cell_height = 8

        self.set_font('arial', 'B', 12)
        for header in dataframe.columns:
            self.cell(cell_width, cell_height, header, 1, 0, 'C')
        self.ln(cell_height)

        self.set_font('arial', '', 12)
        for i in range(0, rows):
            self.set_x(10)
            for header in dataframe.columns:
                value = dataframe[header].ix[i]
                if isinstance(value, float):
                    value = round(value, 4)
                self.cell(cell_width, cell_height, str(value), 1, 0, 'C')
            self.ln(cell_height)
        self.ln(10)


class OverMaxRowsException(Exception):
    pass


def true_positive_rate(true_positives: int,
                       false_negatives: int):
    """
    Returns the true positive rate based on true positives and false negatives
    :param true_positives: number of true positives
    :param false_negatives: number of false negatives
    :return: True Positive Rate
    """
    tpr = true_positives/(true_positives + false_negatives)
    return tpr


def false_positive_rate(false_positives: int,
                        true_negatives: int):
    """
    Returns the false positive rate based on false positives and true negatives
    :param false_positives: number of false positives
    :param true_negatives: number of true negatives
    :return: False Positive Rate
    """
    fpr = false_positives/(false_positives + true_negatives)
    return fpr


def average_odds_ratio(tpr_1, fpr_1, tpr_2, fpr_2):
    """
    Calculates the average odds ratio based on the confusion matrix values of a model prediction
    :param tpr_1: true positive rate
    :param fpr_1: false positives rate
    :param tpr_2: true positive rate
    :param fpr_2: false positives rate
    :return: The Average Odds Ratio
    """
    aor = ((tpr_1 + fpr_1) - (tpr_2 + fpr_2)) / 2
    return aor


def generate_privileged_diff(metrics_table: pd.DataFrame):
    output_table = pd.DataFrame()

    def generate_privileged_diff_metrics(input_data: pd.DataFrame):
        accuracy = input_data.iloc[0, 0] - input_data.iloc[0, 1]
        aod = average_odds_ratio(tpr_1=input_data.iloc[5, 0], fpr_1=input_data.iloc[6, 0],
                                 tpr_2=input_data.iloc[5, 1], fpr_2=input_data.iloc[6, 1])
        output_list = [accuracy, aod]
        return output_list

    output_table['Metrics'] = ['Accuracy', 'AOD']
    output_table['Priv Diff WoW'] = generate_privileged_diff_metrics(metrics_table[['Priv WoW','Not Priv WoW']])
    output_table['Priv Diff WW'] = generate_privileged_diff_metrics(metrics_table[['Priv WW','Not Priv WW']])
    return output_table


def generate_delta_table(metrics_table: pd.DataFrame):
    output_table = pd.DataFrame()

    def generate_delta_metrics(input_data: pd.DataFrame):
        accuracy = input_data.iloc[0, 1] - input_data.iloc[0, 0]
        tn = input_data.iloc[1, 1] - input_data.iloc[1, 0]
        fp = input_data.iloc[2, 1] - input_data.iloc[2, 0]
        fn = input_data.iloc[3, 1] - input_data.iloc[3, 0]
        tp = input_data.iloc[4, 1] - input_data.iloc[4, 0]
        output_list = [accuracy, tn, fp, fn, tp]
        return output_list

    output_table['Metric'] = ['Accuracy', 'True Negative', 'False Positive', 'False Negative', 'True Positive']
    output_table['Delta'] = generate_delta_metrics(metrics_table[['Overall WoW', 'Overall WW']])
    return output_table


def output_probabilities_to_csv(model: sklearn.linear_model, x_test: pd.DataFrame, path: str, priv_group_col, actuals):
    df = pd.DataFrame()
    probs = model.predict_proba(x_test)

    df['probs']=pd.DataFrame(probs)[1]


    temp=pd.DataFrame(priv_group_col)
    df['priv_group_col']=np.array(temp.iloc[:,0])
    df['actuals']=np.array(actuals)

    df.to_csv(path)




