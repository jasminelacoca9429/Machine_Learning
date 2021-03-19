import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json as json
import datetime
from sklearn.preprocessing import MinMaxScaler

'''Q2'''
print('*********************Q2***********************')
raw_data = pd.read_csv('C:/Users/c_jas/Desktop/Python_Project/JHU_COVID-19.csv')
print(raw_data.isnull().sum())
raw_data.drop('Recovered', inplace=True, axis=1)
print(raw_data.head())
print('*********************Q3***********************')

'''Q3 -- Merge same country data'''
dt_by_CnD = raw_data.groupby(['Country/Region', 'Date']).sum()
print('Data by Country and Date', '\n', dt_by_CnD)
print('*********************Q4***********************')


'''Q4 -- Last day data'''
dt_last_day = raw_data[raw_data['Date'] == '2020-12-08']
dt_ld_confirmed_deaths = dt_last_day.groupby(['Date', 'Country/Region']).sum()
print('Last reported day data of total confirmed and deaths in each Country', '\n', dt_ld_confirmed_deaths)

T10_confirmed = dt_ld_confirmed_deaths.nlargest(10, 'Confirmed')
T10_confirmed = T10_confirmed.Confirmed
print('Top 10 Countries of Confirmed Cases:', '\n', T10_confirmed)
T10_deaths = dt_ld_confirmed_deaths.nlargest(10, 'Deaths')
T10_deaths = T10_deaths.Deaths
print('Top 10 Countries of Deaths Cases:', '\n', T10_deaths)
print('*********************Q5***********************')

'''Q5 -- Confirmed cases over time for each country.'''
confirmed_by_CnD = dt_by_CnD.loc[:, 'Confirmed']
print('Confirmed cases over time for each country', '\n', confirmed_by_CnD)

'''Q5 -- Plot a graph of the number of confirmed cases over time for each country'''
plt.figure()
fig, ax = plt.subplots(figsize=(10, 12))
raw_data.groupby(['Date', 'Country/Region'])['Confirmed'].sum().unstack().plot()
plt.title('Number of confirmed cases over time for each country', fontsize=20, fontname='Calibri', c='brown')
plt.xlabel('Time', fontsize=20, fontname='Calibri')
plt.ylabel('Confirmed Cases', fontsize=20, fontname='Calibri')
ax.legend(loc=(1.01, 0.01), ncol=3, title='Country Name',fontsize=6, title_fontsize=10, fancybox=True, framealpha=1, borderpad=1)
plt.show()

print('*********************Q6***********************')

'''Q6 -- Create a bar plot that shows the number of deaths per 100 confirmed cases 
(observed case-fatality ratio) for the 20 most affected countries.'''

dt_by_C = raw_data.groupby('Country/Region').sum()
dt_by_C['OC_Fatality_Ratio'] = dt_by_C.Deaths / (dt_by_C.Confirmed / 100)
Top_20_affected = dt_by_C.nlargest(20, 'OC_Fatality_Ratio').OC_Fatality_Ratio
print('Top 20 Affected Countries ranked by fatality ratio', '\n', Top_20_affected)

ax = Top_20_affected.plot.barh(color='orange')
plt.title('20 Most Affected Countries', fontsize=20, fontname='Calibri',color='brown')
plt.xlabel('Observed Case-Fatality Ratio', fontsize=20, fontname='Calibri')
plt.ylabel('Country', fontsize=20, fontname='Calibri')
plt.gca().invert_yaxis()
for p in ax.patches:
    ax.annotate("%.2f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, -10), textcoords='offset points')
plt.tight_layout()
plt.show()
print('*********************Q7***********************')


'''Q7 -- Compute the ratio between the total number of confirmed cases and the population
size for each country'''

'''Population data import'''
pplt_dt = pd.read_json('C:/Users/c_jas/Desktop/Python_Project/worldpopulation.json')

'''Merge two dataframe by country, reindex'''
confirmed_vs_pplt = pd.merge(dt_by_C, pplt_dt, left_on='Country/Region', right_on='country', left_index=True).set_index(
    'country')
confirmed_vs_pplt = confirmed_vs_pplt.loc[:, ['Confirmed', 'population']]
confirmed_vs_pplt['Confirmed_Cases_PC'] = confirmed_vs_pplt.Confirmed / confirmed_vs_pplt.population
print(confirmed_vs_pplt)
T10_confirmed_pc = confirmed_vs_pplt.nlargest(10, 'Confirmed_Cases_PC').Confirmed_Cases_PC
print('Top 10 countries in confirmed COVID-19 cases per capita:', '\n',
      T10_confirmed_pc)
print('*******************Q8********************')

'''Q8'''
'''Import climate data'''
clmt_dt = pd.read_json('C:/Users/c_jas/Desktop/Python_Project/climate.json')

'''Transfer raw_data into monthly, extract list of 12 months'''
raw_data['Month'] = pd.DatetimeIndex(raw_data['Date']).to_period('M')
confirmed_by_month = raw_data.groupby(['Country/Region', 'Month'], as_index=False)['Confirmed'].sum()
confirmed_by_month['Month'] = confirmed_by_month['Month'].apply(lambda x: x.strftime('%Y-%m'))
m_list = confirmed_by_month.iloc[0:12, 1].tolist()

'''Create climate dataframe'''
new_dt = pd.DataFrame()
for i in range(clmt_dt.shape[0]):
    clmt_by_country = pd.DataFrame(clmt_dt.monthlyAvg[i])
    clmt_by_country['Country/Region'] = clmt_dt.country[i]
    clmt_by_country['Month'] = m_list
    new_dt = pd.concat([new_dt, clmt_by_country], axis=0)

'''Group, choose weather investigation clues: temperature mean for Q8'''
clmt_by_CnM = new_dt.groupby(['Country/Region', 'Month'], as_index=False).mean()
print('Climate data by country and month', '\n', clmt_by_CnM)

'''Merge clmt profile with monthly raw data by country.'''
clmt_vs_confirmed = pd.merge(clmt_by_CnM, confirmed_by_month, how='inner', on=['Country/Region', 'Month'])

'''Add mean temp column in clmt_vs_confirmed'''
clmt_vs_confirmed['ave_temp'] = (clmt_vs_confirmed.high + clmt_vs_confirmed.low) / 2
'''Remove useless columns, for further investigation, keep these 4 only'''
temp_vs_confirmed = clmt_vs_confirmed.loc[:, ['Country/Region', 'Month', 'Confirmed', 'ave_temp']]
print('Average Temperature VS. Confirmed Cases','\n', temp_vs_confirmed)

'''Calculate All Countries' Correlation'''
ct_list = clmt_vs_confirmed.loc[:, 'Country/Region'].unique()
print('Countries:', ct_list)

corr_list = []
for country in ct_list:
    column_1 = temp_vs_confirmed[temp_vs_confirmed['Country/Region'] == country].Confirmed
    column_2 = temp_vs_confirmed[temp_vs_confirmed['Country/Region'] == country].ave_temp
    corr = column_1.corr(column_2)
    corr_list.append(corr)
corr_dic = {'Country': ct_list, 'Correlation': corr_list}
corr_df = pd.DataFrame(corr_dic)
print('Average temperature and confirmed cases amount Correlation Table (A part)', '\n', corr_df)

'''Plot correlation'''
x = corr_df.Country
y = corr_df.Correlation
cc = ['colors'] * len(y)
for n, val in enumerate(y):
    if val < 0:
        cc[n] = 'red'
    else:
        cc[n] = 'green'
plt.title('Correlation of Monthly Confirmed Cases & Average Monthly Temperature', fontsize=20, fontname='Calibri', color='brown')
plt.xlabel('Correlation index', fontsize=20, fontname='Calibri')
plt.ylabel('Country', fontsize=20, fontname='Calibri')
plt.barh(x, y, color=cc)
plt.tight_layout()
plt.show()

'''Q9 -- Select 4 countries, plot respectively.'''

country_list = ['Argentina', 'Italy', 'Japan', 'Australia']

for country in country_list:
    plt.figure()
    fig, ax = plt.subplots(figsize=(14, 12))
    color = 'green'
    ax.set_title(country + ': monthly number of confirmed cases vs. average monthly temperature', fontsize=16, color = 'brown')
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Avg Temp', fontsize=16, color=color)
    ax2 = sns.lineplot(x='Month', y='ave_temp', color=color,
                       data=temp_vs_confirmed[temp_vs_confirmed['Country/Region'] == country])
    ax2 = ax.twinx()
    color = 'red'
    ax2.set_ylabel('confirmed', fontsize=16, color=color)
    ax2 = sns.lineplot(x='Month', y='Confirmed', data=temp_vs_confirmed[temp_vs_confirmed['Country/Region'] == country],
                       sort=False, color=color)
    print(ax2)
    ax2.tick_params(axis='y')
    # plt.show()
    plt.clf()
