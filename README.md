# Yulu-Hypothesis_Testing and Analysis

**Project Overview:**
This project involves analyzing the factors influencing the demand for Yulu's shared electric cycles in the Indian market. By conducting hypothesis testing and exploratory data analysis (EDA), we aim to identify significant variables affecting rental demand and provide actionable business recommendations.

**Introduction:**
Yulu, India's pioneering micro-mobility service provider, has faced recent revenue setbacks and seeks to understand the factors influencing the demand for their shared electric cycles. This analysis will help Yulu tailor their services and strategies to regain profitability and expand their market presence.

**Dataset:**
The dataset includes the following columns:
- datetime: Date and time of the rental
- season: Season (1: spring, 2: summer, 3: fall, 4: winter)
- holiday: Whether the day is a holiday
- workingday: If the day is neither a weekend nor a holiday
- weather: Weather condition (1: Clear, 2: Mist, 3: Light Snow/Rain, 4: Heavy Rain)
- temp: Temperature in Celsius
- atemp: Feels-like temperature in Celsius
- humidity: Humidity level
- windspeed: Wind speed
- casual: Count of casual users
- registered: Count of registered users
- count: Total rental count

## Project Steps:

**Data Preprocessing and EDA:**

- Examine the dataset structure, characteristics, and summary statistics.
- Identify and handle missing values and duplicate records.
- Analyze the distribution of numerical and categorical variables.
- Detect and address outliers using appropriate methods.

**Hypothesis Testing:**

- Conduct various hypothesis tests to determine significant factors affecting rental demand:
  - Two-Sample t-Test: Compare rental counts on weekdays and weekends.
  - Mann-Whitney U test: Compare rental counts on weekdays and weekends (alternate to 2 sample t-test when normality violates)
  - One-Way ANOVA: Compare rental counts across different weather conditions.
  - Chi-Square Test: Assess the relationship between weather conditions and seasons.

**Insights and Business Recommendations:**

- Summarize the findings from the hypothesis tests and EDA.
- Provide actionable business recommendations based on the analysis.
- Key Findings:
  - Rental counts peak during summer and fall seasons, with notable declines in winter and spring.
  - Higher rental activity on working days compared to non-working days.
  - Significantly higher rentals on non-holidays.
  - Majority of rentals occur during clear and cloudy weather, with no rentals during heavy rain.
  - Business Recommendations: Detailed in attached PDF

**Conclusion:**
This project provides a comprehensive analysis of the factors influencing Yulu's rental demand, offering valuable insights and recommendations to boost revenue and improve market strategies.
