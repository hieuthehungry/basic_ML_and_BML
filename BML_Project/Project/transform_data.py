import pandas as pd
import numpy as np

def transform_data(x):
    x['UnknownNumberOfDependents'] = pd.isna(x['NumberOfDependents']).astype(int)
    x['UnknownMonthlyIncome'] = pd.isna(x['MonthlyIncome']).astype(int)

    x['NoDependents'] = (x['NumberOfDependents'] == 0).astype(int)
    x['NoDependents'].loc[pd.isna(x['NoDependents'])] = 0

    x['NumberOfDependents'].loc[x['UnknownNumberOfDependents'] == 1] = 0

    x['NoIncome'] = (x['MonthlyIncome'] == 0).astype(int)
    x['NoIncome'].loc[pd.isna(x['NoIncome'])] = 0

    x['MonthlyIncome'].loc[x['UnknownMonthlyIncome'] == 1] = 0

    x['ZeroDebtRatio'] = (x['DebtRatio'] == 0).astype(int)
    x['UnknownIncomeDebtRatio'] = x['DebtRatio'].astype(int)
    x['UnknownIncomeDebtRatio'].loc[x['UnknownMonthlyIncome'] == 0] = 0
    x['DebtRatio'].loc[x['UnknownMonthlyIncome'] == 1] = 0

    x['WeirdRevolvingUtilization'] = x['RevolvingUtilizationOfUnsecuredLines']
    x['WeirdRevolvingUtilization'].loc[~(np.log(x['RevolvingUtilizationOfUnsecuredLines']) > 3)] = 0
    x['ZeroRevolvingUtilization'] = (x['RevolvingUtilizationOfUnsecuredLines'] == 0).astype(int)
    x['RevolvingUtilizationOfUnsecuredLines'].loc[np.log(x['RevolvingUtilizationOfUnsecuredLines']) > 3] = 0

    x['Log.Debt'] = np.log(np.maximum(x['MonthlyIncome'], np.repeat(1, x.shape[0])) * x['DebtRatio'])
    x['Log.Debt'].loc[~np.isfinite(x['Log.Debt'])] = 0

    x['RevolvingLines'] = x['NumberOfOpenCreditLinesAndLoans'] - x['NumberRealEstateLoansOrLines']

    x['HasRevolvingLines'] = (x['RevolvingLines'] > 0).astype(int)
    x['HasRealEstateLoans'] = (x['NumberRealEstateLoansOrLines'] > 0).astype(int)
    x['HasMultipleRealEstateLoans'] = (x['NumberRealEstateLoansOrLines'] > 2).astype(int)
    x['EligibleSS'] = (x['age'] >= 60).astype(int)
    x['DTIOver33'] = ((x['NoIncome'] == 0) & (x['DebtRatio'] > 0.33)).astype(int)
    x['DTIOver43'] = ((x['NoIncome'] == 0) & (x['DebtRatio'] > 0.43)).astype(int)
    x['DisposableIncome'] = (1 - x['DebtRatio'])*x['MonthlyIncome']
    x['DisposableIncome'].loc[x['NoIncome'] == 1] = 0

    x['RevolvingToRealEstate'] = x['RevolvingLines'] / (1 + x['NumberRealEstateLoansOrLines'])

    x['NumberOfTime30-59DaysPastDueNotWorseLarge'] = (x['NumberOfTime30-59DaysPastDueNotWorse'] > 90).astype(int)
    x['NumberOfTime30-59DaysPastDueNotWorse96'] = (x['NumberOfTime30-59DaysPastDueNotWorse'] == 96).astype(int)
    x['NumberOfTime30-59DaysPastDueNotWorse98'] = (x['NumberOfTime30-59DaysPastDueNotWorse'] == 98).astype(int)
    x['Never30-59DaysPastDueNotWorse'] = (x['NumberOfTime30-59DaysPastDueNotWorse'] == 0).astype(int)
    x['NumberOfTime30-59DaysPastDueNotWorse'].loc[x['NumberOfTime30-59DaysPastDueNotWorse'] > 90] = 0

    x['NumberOfTime60-89DaysPastDueNotWorseLarge'] = (x['NumberOfTime60-89DaysPastDueNotWorse'] > 90).astype(int)
    x['NumberOfTime60-89DaysPastDueNotWorse96'] = (x['NumberOfTime60-89DaysPastDueNotWorse'] == 96).astype(int)
    x['NumberOfTime60-89DaysPastDueNotWorse98'] = (x['NumberOfTime60-89DaysPastDueNotWorse'] == 98).astype(int)
    x['Never60-89DaysPastDueNotWorse'] = (x['NumberOfTime60-89DaysPastDueNotWorse'] == 0).astype(int)
    x['NumberOfTime60-89DaysPastDueNotWorse'].loc[x['NumberOfTime60-89DaysPastDueNotWorse'] > 90] = 0

    x['NumberOfTimes90DaysLateLarge'] = (x['NumberOfTimes90DaysLate'] > 90).astype(int)
    x['NumberOfTimes90DaysLate96'] = (x['NumberOfTimes90DaysLate'] == 96).astype(int)
    x['NumberOfTimes90DaysLate98'] = (x['NumberOfTimes90DaysLate'] == 98).astype(int)
    x['Never90DaysLate'] = (x['NumberOfTimes90DaysLate'] == 0).astype(int)
    x['NumberOfTimes90DaysLate'].loc[x['NumberOfTimes90DaysLate'] > 90] = 0

    x['IncomeDivBy10'] = ((x['MonthlyIncome'] % 10) == 0).astype(int)
    x['IncomeDivBy100'] = ((x['MonthlyIncome'] % 100) == 0).astype(int)
    x['IncomeDivBy1000'] = ((x['MonthlyIncome'] % 1000) == 0).astype(int)
    x['IncomeDivBy5000'] = ((x['MonthlyIncome'] % 5000) == 0).astype(int)
    x['Weird0999Utilization'] = (x['RevolvingUtilizationOfUnsecuredLines'] == 0.9999998999999999).astype(int)
    x['FullUtilization'] = (x['RevolvingUtilizationOfUnsecuredLines'] == 1).astype(int)
    x['ExcessUtilization'] = (x['RevolvingUtilizationOfUnsecuredLines'] > 1).astype(int)

    x['NumberOfTime30-89DaysPastDueNotWorse'] = x['NumberOfTime30-59DaysPastDueNotWorse'] + x['NumberOfTime60-89DaysPastDueNotWorse']
    x['Never30-89DaysPastDueNotWorse'] = x['Never60-89DaysPastDueNotWorse'] * x['Never30-59DaysPastDueNotWorse']

    x['NumberOfTimesPastDue'] = x['NumberOfTime30-59DaysPastDueNotWorse'] + x['NumberOfTime60-89DaysPastDueNotWorse'] + x['NumberOfTimes90DaysLate']
    x['NeverPastDue'] = x['Never90DaysLate'] * x['Never60-89DaysPastDueNotWorse'] * x['Never30-59DaysPastDueNotWorse']
    x['Log.RevolvingUtilizationTimesLines'] = np.log1p(x['RevolvingLines'] * x['RevolvingUtilizationOfUnsecuredLines'])

    x['Log.RevolvingUtilizationOfUnsecuredLines'] = np.log(x['RevolvingUtilizationOfUnsecuredLines'])
    x['Log.RevolvingUtilizationOfUnsecuredLines'].loc[pd.isna(x['Log.RevolvingUtilizationOfUnsecuredLines'])] = 0
    x['Log.RevolvingUtilizationOfUnsecuredLines'].loc[~np.isfinite(x['Log.RevolvingUtilizationOfUnsecuredLines'])] = 0
    x = x.drop('RevolvingUtilizationOfUnsecuredLines', axis=1)

    x['DelinquenciesPerLine'] = x['NumberOfTimesPastDue'] / x['NumberOfOpenCreditLinesAndLoans']
    x['DelinquenciesPerLine'].loc[x['NumberOfOpenCreditLinesAndLoans'] == 0] = 0
    x['MajorDelinquenciesPerLine'] = x['NumberOfTimes90DaysLate'] / x['NumberOfOpenCreditLinesAndLoans']
    x['MajorDelinquenciesPerLine'].loc[x['NumberOfOpenCreditLinesAndLoans'] == 0] = 0
    x['MinorDelinquenciesPerLine'] = x['NumberOfTime30-89DaysPastDueNotWorse'] / x['NumberOfOpenCreditLinesAndLoans']
    x['MinorDelinquenciesPerLine'].loc[x['NumberOfOpenCreditLinesAndLoans'] == 0] = 0

    # Now delinquencies per revolving
    x['DelinquenciesPerRevolvingLine'] = x['NumberOfTimesPastDue'] / x['RevolvingLines']
    x['DelinquenciesPerRevolvingLine'].loc[x['RevolvingLines'] == 0] = 0
    x['MajorDelinquenciesPerRevolvingLine'] = x['NumberOfTimes90DaysLate'] / x['RevolvingLines']
    x['MajorDelinquenciesPerRevolvingLine'].loc[x['RevolvingLines'] == 0] = 0
    x['MinorDelinquenciesPerRevolvingLine'] = x['NumberOfTime30-89DaysPastDueNotWorse'] / x['RevolvingLines']
    x['MinorDelinquenciesPerRevolvingLine'].loc[x['RevolvingLines'] == 0] = 0

    x['Log.DebtPerLine'] = x['Log.Debt'] - np.log1p(x['NumberOfOpenCreditLinesAndLoans'])
    x['Log.DebtPerRealEstateLine'] = x['Log.Debt'] - np.log1p(x['NumberRealEstateLoansOrLines'])
    x['Log.DebtPerPerson'] = x['Log.Debt'] - np.log1p(x['NumberOfDependents'])
    x['RevolvingLinesPerPerson'] = x['RevolvingLines'] / (1 + x['NumberOfDependents'])
    x['RealEstateLoansPerPerson'] = x['NumberRealEstateLoansOrLines'] / (1 + x['NumberOfDependents'])
    x['UnknownNumberOfDependents'] = (x['UnknownNumberOfDependents']).astype(int)
    x['YearsOfAgePerDependent'] = x['age'] / (1 + x['NumberOfDependents'])

    x['Log.MonthlyIncome'] = np.log(x['MonthlyIncome'])
    x['Log.MonthlyIncome'].loc[~np.isfinite(x['Log.MonthlyIncome']) | np.isnan(x['Log.MonthlyIncome'])] = 0
    x = x.drop('MonthlyIncome', axis=1)
    x['Log.IncomePerPerson'] = x['Log.MonthlyIncome'] - np.log1p(x['NumberOfDependents'])
    x['Log.IncomeAge'] = x['Log.MonthlyIncome'] - np.log1p(x['age'])

    x['Log.NumberOfTimesPastDue'] = np.log(x['NumberOfTimesPastDue'])
    x['Log.NumberOfTimesPastDue'].loc[~np.isfinite(x['Log.NumberOfTimesPastDue'])] = 0

    x['Log.NumberOfTimes90DaysLate'] = np.log(x['NumberOfTimes90DaysLate'])
    x['Log.NumberOfTimes90DaysLate'].loc[~np.isfinite(x['Log.NumberOfTimes90DaysLate'])] = 0

    x['Log.NumberOfTime30-59DaysPastDueNotWorse'] = np.log(x['NumberOfTime30-59DaysPastDueNotWorse'])
    x['Log.NumberOfTime30-59DaysPastDueNotWorse'].loc[~np.isfinite(x['Log.NumberOfTime30-59DaysPastDueNotWorse'])] = 0

    x['Log.NumberOfTime60-89DaysPastDueNotWorse'] = np.log(x['NumberOfTime60-89DaysPastDueNotWorse'])
    x['Log.NumberOfTime60-89DaysPastDueNotWorse'].loc[~np.isfinite(x['Log.NumberOfTime60-89DaysPastDueNotWorse'])] = 0

    x['Log.Ratio90to30-59DaysLate'] = x['Log.NumberOfTimes90DaysLate'] - x['Log.NumberOfTime30-59DaysPastDueNotWorse']
    x['Log.Ratio90to60-89DaysLate'] = x['Log.NumberOfTimes90DaysLate'] - x['Log.NumberOfTime60-89DaysPastDueNotWorse']

    x['AnyOpenCreditLinesOrLoans'] = (x['NumberOfOpenCreditLinesAndLoans'] > 0).astype(int)
    x['Log.NumberOfOpenCreditLinesAndLoans'] = np.log(x['NumberOfOpenCreditLinesAndLoans'])
    x['Log.NumberOfOpenCreditLinesAndLoans'].loc[~np.isfinite(x['Log.NumberOfOpenCreditLinesAndLoans'])] = 0
    x['Log.NumberOfOpenCreditLinesAndLoansPerPerson'] = x['Log.NumberOfOpenCreditLinesAndLoans'] - np.log1p(x['NumberOfDependents'])

    x['Has.Dependents'] = (x['NumberOfDependents'] > 0).astype(int)
    x['Log.HouseholdSize'] = np.log1p(x['NumberOfDependents'])
    x = x.drop('NumberOfDependents', axis=1)

    x['Log.DebtRatio'] = np.log(x['DebtRatio'])
    x['Log.DebtRatio'].loc[~np.isfinite(x['Log.DebtRatio'])] = 0
    x = x.drop('DebtRatio', axis=1)

    x['Log.DebtPerDelinquency'] = x['Log.Debt'] - np.log1p(x['NumberOfTimesPastDue'])
    x['Log.DebtPer90DaysLate'] = x['Log.Debt'] - np.log1p(x['NumberOfTimes90DaysLate'])

    x['Log.UnknownIncomeDebtRatio'] = np.log(x['UnknownIncomeDebtRatio'])
    x['Log.UnknownIncomeDebtRatio'].loc[~np.isfinite(x['Log.UnknownIncomeDebtRatio'])] = 0
    # x['IntegralDebtRatio'] = None
    x['Log.UnknownIncomeDebtRatioPerPerson'] = x['Log.UnknownIncomeDebtRatio'] - x['Log.HouseholdSize']
    x['Log.UnknownIncomeDebtRatioPerLine'] = x['Log.UnknownIncomeDebtRatio'] - np.log1p(x['NumberOfOpenCreditLinesAndLoans'])
    x['Log.UnknownIncomeDebtRatioPerRealEstateLine'] = x['Log.UnknownIncomeDebtRatio'] - np.log1p(x['NumberRealEstateLoansOrLines'])
    x['Log.UnknownIncomeDebtRatioPerDelinquency'] = x['Log.UnknownIncomeDebtRatio'] - np.log1p(x['NumberOfTimesPastDue'])
    x['Log.UnknownIncomeDebtRatioPer90DaysLate'] = x['Log.UnknownIncomeDebtRatio'] - np.log1p(x['NumberOfTimes90DaysLate'])

    x['Log.NumberRealEstateLoansOrLines'] = np.log(x['NumberRealEstateLoansOrLines'])
    x['Log.NumberRealEstateLoansOrLines'].loc[~np.isfinite(x['Log.NumberRealEstateLoansOrLines'])] = 0
    x = x.drop('NumberRealEstateLoansOrLines', axis=1)

    x = x.drop('NumberOfOpenCreditLinesAndLoans', axis=1)

    x = x.drop('NumberOfTimesPastDue', axis=1)
    x = x.drop('NumberOfTimes90DaysLate', axis=1)
    x = x.drop('NumberOfTime30-59DaysPastDueNotWorse', axis=1)
    x = x.drop('NumberOfTime60-89DaysPastDueNotWorse', axis=1)

    x['LowAge'] = (x['age'] < 18) * 1
    x['Log.age'] = np.log(x['age'] - 17)
    x['Log.age'].loc[x['LowAge'] == 1] = 0
    x = x.drop('age', axis=1)
    return x