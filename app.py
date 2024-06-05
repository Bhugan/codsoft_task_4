import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title('Sales Price Prediction Model')

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the dataset
    st.header('Dataset')
    st.write(df.head())

    # Display basic EDA
    st.header('Exploratory Data Analysis')

    # Show dataset statistics
    if st.checkbox('Show dataset statistics'):
        st.write(df.describe())

    # Check for missing values
    if st.checkbox('Show missing values'):
        st.write(df.isna().sum())

    # Pairplot
    if st.checkbox('Show pairplot'):
        sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg')
        st.pyplot(plt)

    st.header('Sales Prediction Model')

    # Define the dependent and independent variables
    y = df['Sales']
    X = df[['TV', 'Radio', 'Newspaper']]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add a constant to the independent variables
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Fit the model
    model = sm.OLS(y_train, X_train).fit()

    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute and display the accuracy (mean squared error)
    mse = mean_squared_error(y_test, y_pred)
    st.subheader('Model Accuracy')
    st.write(f'Mean Squared Error: {mse:.2f}')

    # Pairplot with the fitted line
    if st.checkbox('Show pairplot with fitted line'):
        sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg')
        st.pyplot(plt)

    # User input for new predictions
    st.header('Make a Prediction')
    tv_budget = st.number_input('Enter TV Budget:', min_value=0.0, step=0.1)
    radio_budget = st.number_input('Enter Radio Budget:', min_value=0.0, step=0.1)
    newspaper_budget = st.number_input('Enter Newspaper Budget:', min_value=0.0, step=0.1)

    if st.button('Predict'):
        new_data = pd.DataFrame({'const': [1], 'TV': [tv_budget], 'Radio': [radio_budget], 'Newspaper': [newspaper_budget]})
        prediction = model.predict(new_data)
        st.write(f'Predicted Sales: {prediction[0]:.2f}')

else:
    st.info('Please upload a CSV file to get started.')
