#!/usr/bin/env python
# coding: utf-8

# # Business Case: Yulu - Hypothesis Testing

# **About Yulu**
# 
# Yulu is India’s leading micro-mobility service provider, which offers unique vehicles for the daily commute. Starting off as a mission to eliminate traffic congestion in India, Yulu provides the safest commute solution through a user-friendly mobile app to enable shared, solo and sustainable commuting.
# 
# Yulu zones are located at all the appropriate locations (including metro stations, bus stands, office spaces, residential areas, corporate offices, etc) to make those first and last miles smooth, affordable, and convenient!
# 
# Yulu has recently suffered considerable dips in its revenues. They have contracted a consulting company to understand the factors on which the demand for these shared electric cycles depends. Specifically, they want to understand the factors affecting the demand for these shared electric cycles in the Indian market.

# **Column Profiling:**
# 
# - datetime: datetime
# - season: season (1: spring, 2: summer, 3: fall, 4: winter)
# - holiday: whether day is a holiday or not
# - workingday: if day is neither weekend nor holiday is 1, otherwise is 0.
# - weather:
#     - 1 - Clear, Few clouds, partly cloudy, partly cloudy
#     - 2 - Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     - 3 - Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     - 4 - Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp: temperature in Celsius
# - atemp: feeling temperature in Celsius
# - humidity: humidity
# - windspeed: wind speed
# - casual: count of casual users
# - registered: count of registered users
# - count: count of total rental bikes including both casual and registered

# In[136]:


import pandas as pd
import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt
import seaborn as sns


# In[137]:


data = pd.read_csv(r"C:\Users\n.rahman\OneDrive - BALADNA\Desktop\BALADNA\Ex Docs\SCALER-DSML\Module 7 -Statistics\bike_sharing.csv")
data.sample(5)


# ## Exploratory Data Analysis

# In[138]:


data.shape


# In[139]:


data.info()


# In[140]:


data.describe()


# ### Missing Values & Duplicates Check

# In[141]:


missing_value = pd.DataFrame({"Missing_Values":data.isnull().sum(),"Percentage":(data.isnull().sum()/len(data))*100})
missing_value


# In[269]:


data.duplicated().sum() #no duplicates


# ### Datatype conversions of attributes

# In[142]:


data["season"].value_counts()


# In[143]:


data["holiday"].value_counts()


# In[144]:


data["workingday"].value_counts()


# In[145]:


data["weather"].value_counts()


# In[146]:


data["datetime"] = pd.to_datetime(data["datetime"])
data["season"] = data["season"].astype("category")
data["holiday"] = data["holiday"].astype("category")
data["workingday"] = data["workingday"].astype("category")
data["weather"] = data["weather"].astype("category")


# In[147]:


data.info()


# ### Univariate Analysis

# In[148]:


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
sns.countplot(data=data,x="season", ax=axs[0,0])
sns.countplot(data=data,x="holiday",ax = axs[0,1])
sns.countplot(data=data,x="workingday",ax=axs[1,0])
sns.countplot(data=data,x="weather",ax=axs[1,1])
plt.show()


# **Observations**
# 
# - For all four seasons, the number of transactions seems to be nearly equal; however, we need to examine the rental counts and impacts.
# - During holidays, rentals are minimal compared to non-holidays, which is expected.
# - On working days, rental transactions are higher compared to non-holidays, which is logical.
# - Weather count plots indicate that most rentals occur during clear and cloudy weather (1) compared to other weather periods, while during heavy rain (4), only one rental occurred. So definitely there seems to be some patterns in rentals during weather.

# ### Bivariate Analysis

# In[150]:


fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(16,18))

#first row
sns.boxplot(x=data["season"],y=data["count"], ax = axs[0,0]).set_title("Distribution of Rentals by Season")
sns.barplot(x="season",y="count",data=data.groupby("season")["count"].sum().reset_index(), ax=axs[0,1]).set_title("Rentals Count by Season")

#second row
sns.boxplot(x=data["workingday"],y=data["count"], ax=axs[1,0]).set_title("Distribution of Rentals by Working Day")
sns.barplot(x="workingday",y="count",data=data.groupby("workingday")["count"].sum().reset_index(), ax=axs[1,1]).set_title("Rentals Count by Working Day")


#third row
sns.boxplot(x=data["holiday"],y=data["count"], ax=axs[2,0]).set_title("Distribution of Rentals by Holiday")
sns.barplot(x="holiday",y="count",data=data.groupby("holiday")["count"].sum().reset_index(), ax=axs[2,1]).set_title("Rentals Count by Holiday")

#fourth row
sns.boxplot(x=data["weather"],y=data["count"], ax=axs[3,0]).set_title("Distribution of Rentals by Weather")
sns.barplot(x="weather",y="count",data=data.groupby("weather")["count"].sum().reset_index(), ax=axs[3,1]).set_title("Rentals Count by Weather")

fig.subplots_adjust(hspace=0.5)
plt.show()


# **Observations**
# 
# - From the season boxplot and bar chart, it is evident that rental counts peak during summer and the fall season and starts drop during winter with a notable decline in spring as well. Each season exhibits outliers that requires further investigation.
# - The working day plot clearly indicates higher rental activity on working days compared to non-working days. The boxplot shows a greater number of outliers on working days.
# - Rental activity is significantly higher on non-holidays which is expected and the data contains numerous outliers that need further examination.
# - Analysis of weather conditions reveals that rentals are absent during heavy rain, with outliers present in all other weather conditions. The majority of rentals occur during clear and cloudy weather, highlighting the substantial influence of weather and season on Yulu rental's sales and revenue.

# ### Correlation between features

# In[375]:


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

#first row
sns.scatterplot(data=data,x = data["temp"],y= data["count"],ax=axs[0,0])
sns.scatterplot(data=data,x = data["atemp"],y= data["count"],ax=axs[0,1])
sns.scatterplot(data=data,x = data["humidity"],y= data["count"],ax=axs[0,2])

#second row
sns.scatterplot(data=data,x = data["windspeed"],y= data["count"],ax=axs[1,0])
sns.scatterplot(data=data,x = data["casual"],y= data["count"],ax=axs[1,1])
sns.scatterplot(data=data,x = data["registered"],y= data["count"],ax=axs[1,2])

fig.subplots_adjust(hspace=0.5)
plt.show()




# In[391]:


cr = data[["temp","atemp","humidity","windspeed","casual","registered","count"]]
cr.corr(method = "spearman")


# In[398]:


plt.figure(figsize=(10,6))
sns.heatmap(cr.corr(method="spearman"),annot=True, cmap="coolwarm",linewidth=0.5)


# **Observations**
# 
# - Positive Correlation: the correlation between rental_count and registered users is 0.99, it means that as the registered useres increases, the rental count also tends to increase.they follow a linear relation.
# - Negative Correlation: the correlation between rental_count and humidity is slighlty negative correlated -0.35, it means that as the humidity increases, the rental count tends to decrease.
# - Weak/No Correlation: Other features suggests that there's no strong linear relationship with rental counts

# ### Rentals by month/ year

# In[151]:


data["datetime"].min()


# In[152]:


data["datetime"].max()


# In[153]:


data["year"] = data["datetime"].dt.year
data["month"] = data["datetime"].dt.month


# In[154]:


monthly_rentals = data.groupby(["year","month"])["count"].sum().reset_index()


# In[155]:


plt.figure(figsize=(9,4))
sns.lineplot(data=monthly_rentals,x="month",y="count",hue="year",marker='o', markersize=8)
plt.title('Monthly Rental Counts by Month/Year')
plt.xlabel('Month')
plt.ylabel('Rental Counts')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.legend(title='Year')
plt.show()


# **Observation**
# 
# - The timeline line chart reveals a clear monthly rental pattern over the two-year period.
# - Although the number of rentals varies significantly, we focus on the overall trend in rental activity over the months rather than the exact counts.
# - For both 2011 and 2012, the rental trends follow a consistent pattern from November to June, despite differences in rental numbers. Rentals decreased from July to November in 2011, whereas they increased during the same period in 2012, with a slight drop in July.May be more users have rented during this period picking up the same. Requires more data or further analysis.
# - The months with rental drops correspond possibly to winter, with rentals starting to increase in spring and peaking in summer.

# ## Outliers Detection

# In[156]:


data.describe(include="all")


# #### Season

# In[157]:


data["season"].unique()


# In[248]:


for i in data["season"].unique():
    season_data = data.loc[data["season"] == i]["count"].reset_index()
    Q1 = np.percentile(season_data["count"],25)
    Q3 = np.percentile(season_data["count"],75)
    IQR = Q3-Q1
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = np.round(len(season_data.loc[season_data["count"]>upper_bound])/len(season_data)*100,2)
    
    print("Season-",i,"25th Percentile=",Q1 ,"75th Percentile=",Q3,"IQR=",IQR,"Upper_Whisker=",upper_bound,"% of Outliers=",outliers)
   
    


# #### Working Day

# In[250]:


data["workingday"].unique()


# In[252]:


for i in data["workingday"].unique():
    wd_data = data.loc[data["workingday"] == i]["count"].reset_index()
    Q1 = np.percentile(wd_data["count"],25)
    Q3 = np.percentile(wd_data["count"],75)
    IQR = Q3-Q1
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = np.round(len(wd_data.loc[wd_data["count"]>upper_bound])/len(wd_data)*100,2)
    
    print("Working Day-",i, "25th Percentile=",Q1 ,"75th Percentile=",Q3,"IQR=",IQR,"Upper_Whisker=",upper_bound,"% of Outliers=",outliers)
   
    


# #### Holiday

# In[254]:


data["holiday"].unique()


# In[255]:


for i in data["holiday"].unique():
    hd_data = data.loc[data["holiday"] == i]["count"].reset_index()
    Q1 = np.percentile(hd_data["count"],25)
    Q3 = np.percentile(hd_data["count"],75)
    IQR = Q3-Q1
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = np.round(len(hd_data.loc[hd_data["count"]>upper_bound])/len(hd_data)*100,2)
    
    print("Holiday-",i, "25th Percentile=",Q1 ,"75th Percentile=",Q3,"IQR=",IQR,"Upper_Whisker=",upper_bound,"% of Outliers=",outliers)
   


# #### Weather

# In[257]:


data["weather"].unique()


# In[259]:


for i in data["weather"].unique():
    we_data = data.loc[data["weather"] == i]["count"].reset_index()
    Q1 = np.percentile(we_data["count"],25)
    Q3 = np.percentile(we_data["count"],75)
    IQR = Q3-Q1
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = np.round(len(we_data.loc[we_data["count"]>upper_bound])/len(we_data)*100,2)
    
    print("Weather-",i, "25th Percentile=",Q1 ,"75th Percentile=",Q3,"IQR=",IQR,"Upper_Whisker=",upper_bound,"% of Outliers=",outliers)
   


# **Observations**
# 
# - For this dataset I will retain all the outliers among all the attributes as outliers can contain valuable information and their removal can sometimes lead to misleading or inaccurate results in analysis. 

# ### Test for Normality

# In[356]:


from statsmodels.graphics.gofplots import qqplot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

#first row
sns.histplot(data["count"], ax = axs[0]).set_title("Distribution of Rentals")
qqplot(data["count"],line="s", ax=axs[1])


# **Observation** - In short, rental counts are not normally distributed, its right skewed or positve skewed. Evident from QQplot as well its non-normally distributed. 

# # Hypothesis Testing

# In[400]:


#importing libraries

from scipy.stats import ttest_ind,chisquare,chi2,f_oneway,mannwhitneyu,chi2_contingency


# ### To check if Working Day has an effect on the number of electric cycles rented 

# #### Using 2 Sample T-test

# In[401]:


df = data[["workingday","count"]]
df.groupby("workingday")["count"].mean()


# #framing the hypothesis
# 
# - H0: There is no significant difference in rentals between the working & non working days
# - Ha: There is significant difference in rentals between the working & non working days

# In[402]:


df_working = df.loc[df["workingday"]==1]["count"]
df_notworking = df.loc[df["workingday"]==0]["count"]


# In[430]:


alpha =0.05 #significance value


# In[431]:


#appylying 2 sample t-test
t_stat, pvalue = ttest_ind(df_working, df_notworking, alternative = "two-sided")
t_stat,pvalue


# In[433]:


if pvalue<alpha:
    print("Reject the null hypothesis, There is significant difference in mean rentals between working and non-working days")
else:
    print("Fail to Reject the null hypothesis, There is no significant difference in mean rentals between working and non-working days")
    print("We dont have sufficient evidence to say that working day had impact on bike rentals")


# #### Using Mann-Whitney U test as the assumption of normality is violated to use 2 sample t-test

# In[406]:


stat, p_value = mannwhitneyu(df_working,df_notworking)
stat,p_value


# In[407]:


if p_value<alpha:
    print("Reject the null hypothesis, There is significant difference in rentals between working and non-working days")
else:
    print("Fail to Reject the null hypothesis, There is no significant difference in rentals between working and non-working days")
    print("We dont have sufficient evidence to say that working day had impact on bike rentals")


# In[302]:


data.groupby("workingday")["count"].sum()


# In[303]:


data.groupby("workingday")["count"].mean()


# ### To check if No. of cycles rented is similar or different in different weather conditions using One Way ANOVA test

# #framing the hypothesis
# 
# - H0: No difference in cycle rentals during different weather conditions
# - Ha: There is differrnce in cycle rentals during difference weather conditions

# In[307]:


data["weather"].value_counts()


# In[319]:


w1 = data.loc[data["weather"]==1]["count"]
w2 = data.loc[data["weather"]==2]["count"]
w3 = data.loc[data["weather"]==3]["count"]
w4 = data.loc[data["weather"]==4]["count"]


# In[323]:


f_stats, p_value = f_oneway(w1,w2,w3,w4)
f_stats,p_value


# In[326]:


alpha =0.05
if p_value<alpha:
    print("Reject the null hypothesis, There is significant difference in rentals on different weather conditions")
else:
    print("Fail to Reject the null hypothesis, There is no significant difference in rentals on different weather conditions")


# ### To check if No. of cycles rented is similar or different in different  seasons using One Way ANOVA test

# In[327]:


s1 = data.loc[data["season"]==1]["count"]
s2 = data.loc[data["season"]==2]["count"]
s3 = data.loc[data["season"]==3]["count"]
s4 = data.loc[data["season"]==4]["count"]


# In[342]:


#checking the assumptions

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12,10))

#first row
sns.histplot(s1, ax = axs[0,0]).set_title("Distribution of Rentals for Spring Season")
qqplot(s1,line="s", ax=axs[0,1])


#second row
sns.histplot(s2, ax = axs[1,0]).set_title("Distribution of Rentals for Summer Season")
qqplot(s2,line="s", ax=axs[1,1])


#third row
sns.histplot(s3, ax = axs[2,0]).set_title("Distribution of Rentals for Fall Season")
qqplot(s3,line="s", ax=axs[2,1])

#fourth row
sns.histplot(s4, ax = axs[3,0]).set_title("Distribution of Rentals for Winter Season")
qqplot(s4,line="s", ax=axs[3,1])

fig.subplots_adjust(hspace=0.5)
plt.show()


# In[328]:


f_stats, p_value = f_oneway(s1,s2,s3,s4)
f_stats,p_value


# In[329]:


alpha =0.05
if p_value<alpha:
    print("Reject the null hypothesis, There is significant difference in rentals on different seasons")
else:
    print("Fail to Reject the null hypothesis, There is no significant difference in rentals on different seasons")


# ### Check if the Weather conditions are significantly different during different Seasons?

# #### Using Chi-Square Contingency Test

# #framing the hypothesis
# 
# - H0: Weather and Season are independent
# - Ha: Weather and Season are not independent

# In[361]:


crosstab = pd.crosstab(data["weather"],data["season"])
crosstab


# In[364]:


chi_stat, p_value, df, exp_freq = chi2_contingency(crosstab) # chi_stat, p_value, df, expected value

print("chi_stat:",chi_stat)
print("p_value:",p_value)
print("df:",df)
print("exp_freq:",exp_freq)


# In[365]:


alpha = 0.05

if p_value < alpha:
    print("Reject H0")
    print("Weather and Season are dependent")
else:
    print("Fail to reject H0")
    print("Weather and Season are not dependent")


# **Recommendations**
# 
# - Introduce special promotions and discounts during summer and fall when rentals peak to capitalize on high demand and further increase revenue.
# - Develop targeted marketing campaigns for winter months to mitigate the drop in rentals. Offer winter deals such as discounted long term rentals and accessories like disposible rain coats. 
# - Given the higher rental activity on working days, consider introducing weekday specific packages or subscriptions for daily commuters to boost regular usage.
# - Make dynamic pricing that adjusts based on weather conditions, offering discounts on clear and cloudy days to maximize rentals.
# - Will have to conduct a detailed analysis of outliers to understand unusual rental behaviors and identify opportunities service improvements. Need more data like demographics, city etc. 
# - Enhance the customer experience by ensuring bikes are well-maintained, easily accessible, and equipped with necessary accessories e.g. helmets, phone holders, phone charger.
# - Use the insights from the timeline line chart to plan and allocate resources effectively throughout the year, ensuring adequate bike availability and operational support during peak and off-peak months.
# - Develop and deploy a machine learning model to predict rental demand based on historical data, weather conditions, seasonal trends, holidays, and other relevant factors.
# - Make deal with food delivery partners to make use of Yulu bikes equipped with all necessary accessories to preserve the food quality and delivery on time.
# 
# 
# 
# 

# In[ ]:




