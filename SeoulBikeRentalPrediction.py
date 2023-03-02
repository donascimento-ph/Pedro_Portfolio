import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from scipy.stats import f_oneway
from scipy import stats
import altair as alt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
import math
from sklearn.ensemble import RandomForestRegressor

st.title("Seoul Bike Rental Prediction")
st.subheader("By: Pedro H. do Nascimento")
st.markdown("With a data set that provides us with weather information, we can forecast the demand for rental bikes at certain times of the day. Demand forecasting is important to ensure that there is a sufficient number of bikes available for users, avoiding queues and ensuring that mobility needs are satisfactorily met. This foresight can help the company better manage its resources and make more effective decisions regarding the supply and maintenance of rental bikes. In addition, the forecast can be used to plan bike routes and maintenance schedules, as well as marketing and promotion strategies to attract more users during low demand times.")

st.header("Solution planning")
st.markdown("**Data Collection:** Collecting data on bicycle use in a city, including information about the weather and other variables that may influence demand for bikes.")
st.markdown("**Data Cleanup:** Clean up data to remove missing values, duplicate data, and other errors. It is also important to ensure that the data is in a format suitable for analysis.")
st.markdown("**Feature Selection:** Select the most relevant features for the analysis by using methods such as correlation analysis and ANOVA test.")
st.markdown("**Exploratory Data Analysis:** Perform exploratory data analysis to understand data distribution, identify patterns and trends, and assess the relationship between different variables.")
st.markdown("**Data Modeling:** Choose a suitable Machine Learning model to predict demand for bikes based on available variables. A regression model, for example, can be used to predict the number of bikes rented on a given day based on variables such as temperature, humidity and wind speed.")
st.markdown("**Machine Learning Algorithms Training:** Train the Machine Learning model with available data and adjust its parameters to improve prediction accuracy.")
st.markdown("**Algorithm Performance Evaluation:** Evaluate the performance of the Machine Learning model using metrics such as Mean Absolute Error (MAE) or Mean Squared Error (MSE).")
st.markdown("**Translation of performance into financial results:** Translate model performance into financial results to determine the financial gain if the solution were implemented now. For example, if the model can predict the demand for bikes accurately enough, the bike rental company can adjust the amount of bikes available in real time to meet market demand, avoiding overstocking or shortages of bikes.")
st.markdown("**Publishing the model to production:** Deploying the machine learning model to a production environment, allowing it to be used in real-time to predict demand for bikes.")

df = pd.read_csv('bikerental.csv')
df_test=pd.read_csv('bikerental_test.csv')

st.header('Data Collection')

st.markdown("The dataset below presents data about bike usage in Seoul, including information about the weather and other variables that could influence bike demand.")

st.write(df)

st.header('Data Cleanup')

st.markdown("In this step we will check for missing values, duplicate data and other errors. It is also important to ensure that the data is in a format suitable for analysis.")

datasetoverview = Image.open('datasetoverview.jpg')

st.image(datasetoverview, caption='Dataset Overview')

st.markdown("The dataset statistics are in compliance for a good analysis, however it is necessary to verify that the variable types are in the correct format.")

dfinfo = Image.open('dfinfo.jpg')

st.image(dfinfo, caption='Dataset Info')

st.markdown("The first action taken was to convert the separate **Date** column to **Year**, **Month**, **Day** and **DayOfWeek**. This action allows our models to consider this information separately and, in this way, find possible relationships between them and the target variable.")
st.markdown("In addition, I removed the **ID** column from our dataset, as it does not provide relevant information for predicting the number of rented bikes.")
st.markdown("Finally, I perform a transformation on the target variable **Bikes_Rented** using the square root. This action is known as a Box-Cox transformation and is common to reduce variation and improve data distribution. This way, our models can work with a smoother and more excitingly distributed target variable, which can lead to better forecasting results.")

df = df.drop(['Id'], axis=1)
df_test = df_test.drop(['Id'], axis=1)

t=[datetime.strptime(i, '%d/%m/%Y')for i in df['Date']]
df['Date']=t
t2=[datetime.strptime(i, '%d/%m/%Y')for i in df_test['Date']]
df_test['Date']=t2


df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df["year"]=df["Date"].dt.year
df['dow'] = df['Date'].dt.dayofweek

df_test['month'] = df_test['Date'].dt.month
df_test['day'] = df_test['Date'].dt.day
df_test["year"]=df_test["Date"].dt.year
df_test['dow'] = df_test['Date'].dt.dayofweek
df.drop(['Date'], inplace= True, axis = 1)
df_test.drop(['Date'], inplace= True, axis = 1)
year_encode={2017:0,2018:1}
df['year']=df["year"].map(year_encode)
df_test['year']=df_test["year"].map(year_encode)

st.header("Feature Selection")

st.markdown("**Numerical Features**")

st.markdown("Let's take a look at the correlation heatmap.")
st.set_option('deprecation.showPyplotGlobalUse', False)
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
st.pyplot()

st.markdown("Temperature is highlighted on this map. Although the correlation is not as strong with the target variable, it is possible to observe that the number of rented bikes has a positive correlation with temperature. It can be understood that cycling in cold weather can be unpleasant, and therefore a rise in temperature would increase the demand for bikes. It is important to keep all these numerical variables.")

st.markdown("**Categorical Features**")

st.markdown("Before transforming the categorical variables into dummy variables, I want to see the importance of each one in explaining the variation in the number of rented bikes")

st.markdown("I need to test if categorical features have influence on a continuous variable that is my target, ANOVA is an easiest statistical filtering method I can use. Basically, it helps you determine if there are significant differences between the means of groups made up of different categories of your categorical variable versus your continuous variable.")

seasons = df['Season'].unique()
data = {season: df['Bikes_Rented'][df['Season'] == season] for season in seasons}
fvalue, pvalue = stats.f_oneway(*data.values())

st.write("### ANOVA Test - Season vs Bike Rental Count")
st.write(f"**F-value: {fvalue:.2f}**")
st.write(f"**P-value: {pvalue:.4f}**") 

st.markdown("The result of the ANOVA Test for the variable Season versus the number of bike rentals (Bike Rental Count) showed an F-value of 659.79 and a P-value of 0.0000.")
st.markdown("The F-value indicates the relationship between the variability of the groups (in this case, the seasons of the year) and the variability within the groups. The higher the F value, the greater the difference between the means of the groups and the smaller the variability within groups. In this case, the F-value is quite high, which suggests that the seasons have a significant effect on the number of bike rentals.")
st.markdown("The P-value indicates the probability of obtaining an F-value as high as the observed one, assuming that the means of the groups are equal (that is, that the seasons do not affect the number of bicycle rentals). The smaller the value of P, the less likely that this hypothesis is true. In this case, the value of P is very low (0.0000), which suggests that the null hypothesis (that the seasons do not affect the number of bicycle rentals) can be rejected with a high degree of confidence. That is, there is strong statistical evidence that the seasons affect the number of bike rentals.")

st.header("Exploratory Data Analysis")

st.markdown("**Does bike rental demand vary by time of day? Is there a higher demand during the morning and afternoon rush hour? Does demand decrease overnight?**")

hourly_rentals = df.groupby('Hour')['Bikes_Rented'].mean().reset_index()

line_chart = alt.Chart(hourly_rentals).mark_line().encode(
    x='Hour',
    y='Bikes_Rented'
).properties(
    width=600,
    height=400,
)
st.altair_chart(line_chart)

st.markdown("We can see from the graph that there is a variation in bike rental demand throughout the day, with demand peaking during the morning and afternoon rush hours. During the night, the demand decreases considerably. Therefore, we can say that there is a relationship between the time of day and the demand for bicycle rentals.")

st.markdown("**Does the weather have a significant impact on bike rental demand? Are people less likely to rent bikes on rainy days or in extreme temperatures?**")

dfG2 = pd.DataFrame({'Temperature': df['Temperature'], 'Rainfall': df['Rainfall'], 'Bikes_Rented': df['Bikes_Rented']})

fig = px.scatter(df, x='Temperature', y='Bikes_Rented', color='Rainfall', hover_data=['Temperature', 'Bikes_Rented'])

st.plotly_chart(fig)

st.markdown("Cycling in warmer weather can be more comfortable and enjoyable for many people. Warm weather can make leisure time more enjoyable and inviting for outdoor activities such as cycling. Also, people can save money on fuel by choosing to ride a bike instead of driving a car on hot days. All these factors can lead to a significant increase in the demand for bicycles.")

st.header("Data Modeling")

st.markdown("Based on the available variables, a suitable machine learning model for predicting demand for bikes could be a regression model such as multiple linear regression or decision tree regression.")
st.markdown("Multiple linear regression is a simple and widely used method for predicting a continuous response variable (in this case, the demand for bikes) based on multiple predictor variables (such as temperature, rainfall, hour of the day, day of the week, etc.). This model assumes a linear relationship between the predictors and the response variable.")
st.markdown("Decision tree regression is a non-parametric model that can handle both categorical and continuous predictor variables. It recursively partitions the data into subsets based on the predictor variables to create a decision tree. Each leaf node represents a prediction for the response variable. This model can capture non-linear relationships between the predictors and the response variable.")
st.markdown("Both of these models can be trained using historical data on bike rentals, and then used to predict the demand for bikes based on the values of the predictor variables for future periods. The choice of model ultimately depends on the characteristics of the data and the specific requirements of the prediction task.")

st.header("Machine Learning Algorithms Training and Algorithm Performance Evaluation")

dummies = pd.get_dummies(df['Season'], prefix='Season')
df = pd.concat([df, dummies], axis=1)
dummies = pd.get_dummies(df_test['Season'], prefix='Season')
df_test = pd.concat([df_test, dummies], axis=1)
df.drop('Season', axis=1, inplace=True)
df_test.drop('Season', axis=1, inplace=True)

df['Bikes_Rented']=round(np.sqrt(df['Bikes_Rented']),1)
df['Wind_Speed']=round(np.sqrt(df['Wind_Speed']),1)

st.markdown("**Multiple Linear Regression**")

X = df.drop('Bikes_Rented', axis=1)
y = df['Bikes_Rented']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("MAE: 5.34")
st.markdown("MSE: 48.75")
st.markdown("R2 Score: 0.68")

st.markdown("**Decision Tree Regressor**")

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("MAE: 2.49")
st.markdown("MSE: 16.03")
st.markdown("R2 Score: 0.89")

st.markdown("**Random Forest Regressor**")

valores_padrao = X.mean()

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("MAE:  1.74")
st.markdown("MSE:  7.55")
st.markdown("R2 Score:  0.95")

st.markdown("The results show that the regression models performed increasingly better, with the Random Forest Regressor performing the best, with a MAE of 1.74, MSE of 7.55 and an R2 Score of 0.95. These values indicate that Random Forest Regressor was able to accurately predict the number of bikes rented based on the variables in the dataset. Linear regression and Decision Tree Regressor models showed intermediate performances, with R2 Scores of 0.68 and 0.89, respectively. In general, these results indicate that regression models are able to predict the number of bikes rented with good accuracy based on the variables available in the dataset.")

st.markdown("**Seoul Bike Rental Predictor**")

features = st.multiselect("Select the variables to predict bike rental:", list(X.columns))

inputs = valores_padrao.copy()
for feature in features:
    value = st.slider(f"Value to {feature}:", float(X[feature].min()), float(X[feature].max()), float(valores_padrao[feature]))
    inputs[feature] = value

bikerental = rf.predict([inputs])

st.markdown(f"Bike Rental Predict:**{bikerental}**")

st.markdown("The Random Forest Regressor may have fared better than the other algorithms because of its ability to handle a large number of explanatory variables and to identify complex relationships between variables.")

st.header("Translation of performance into financial results")

st.markdown("Demand forecasting plays a crucial role in ensuring that the bike rental company meets the mobility needs of its users by providing a sufficient number of bikes. By accurately predicting the demand for bikes, the company can avoid shortages or overstocking of bikes, which can result in dissatisfied customers and lost revenue.")
st.markdown("Moreover, demand forecasting can assist the company in better managing its resources and making more effective decisions related to the supply and maintenance of rental bikes. The forecast can help the company plan maintenance schedules, ensuring that the bikes are always in good condition and available for use.")
st.markdown("In addition, the demand forecast can be used to devise marketing and promotion strategies to attract more users during low demand times, increasing revenue and improving the overall user experience. Overall, accurate demand forecasting can help the bike rental company improve its operations and financial performance, while also providing better service to its customers.")

