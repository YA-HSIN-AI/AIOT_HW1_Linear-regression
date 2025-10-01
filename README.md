# AIOT_HW1_Linear-regression
write python to solve simple linear regression problem, following CRISP-DM steps
1. Business Understanding

Define project objectives and requirements.

Establish success criteria (e.g., predicting stock price trends, providing clear visualization).

Align with the course or project learning goals.

In this project:

The goal is to build a simple regression and visualization tool that downloads financial data and fits a linear regression model.

Success means: the Streamlit app can fetch stock data, fit a regression line, plot results, and allow CSV export.

2. Data Understanding

Collect, describe, and explore the data.

Detect quality issues, missing values, and anomalies.

Understand the data format, types, and distributions.

In this project:

Data is collected using yfinance (e.g., AMZN, TSMC).

Fields include: Date, Open, High, Low, Close, Volume.

Data is explored with Pandas and visualized using Streamlit.

3. Data Preparation

Clean data (handle missing values, format timestamps).

Select features and engineer new features if necessary.

Transform data into a structure suitable for modeling.

In this project:

The Close price is selected as the main target variable.

Data can be resampled (daily, weekly, monthly).

Limit the number of data points to ensure performance.

4. Modeling

Choose and apply appropriate models (e.g., regression, classification).

Adjust model parameters.

Ensure the model is suitable for the prepared data.

In this project:

Apply Scikit-Learn’s LinearRegression.

Model stock price (Close) against time index.

Plot both actual data and regression line for comparison.

5. Evaluation

Validate model performance.

Check whether the model meets project objectives.

Identify model limitations and improvement opportunities.

In this project:

Evaluate whether the regression line follows the trend.

Discuss limitations (linear regression is too simple for real stock forecasting).

Conclusion: The app is useful for demonstration and learning, but not suitable for financial decision-making.

6. Deployment

Deploy into a usable environment.

Provide visualization, API, or web application.

Include documentation and usage instructions.

In this project:

Built an interactive web app with Streamlit.

Supports changing ticker symbol, date range, and resampling frequency.

Includes CSV export functionality.

Deployed to Streamlit Cloud so external users can directly access via a web link.
