import pandas as pd
import fairness as fr
import os

df = pd.read_csv(os.path.join(os.pardir, 'data', 'finaldata.csv'))

scores_path = os.path.join(os.pardir, 'data', 'logreg_scores.csv')
scores = pd.read_csv(scores_path)

example = fr.Fairness(input_dataframe=df,
                      label_names=['salary'],
                      protected_attribute_names=['sex_ Male'],
                      privileged_groups=[{'sex_ Male': 1}], #Male
                      unprivileged_groups = [{'sex_ Male': 0}]  #Female
                      )

if __name__ == '__main__':

    example.generate_pdf_report(yvar='salary',
                                protected_variable='sex_ Male',
                                scores=scores['0'],
                                false_positive_cost=1000,
                                false_negative_cost=500,
                                filepath='Salary_Prediction_Fairness.pdf')