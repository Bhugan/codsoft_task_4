import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

    # Scatter plot of TV and Sales with fitted line
    st.header('Sales Prediction Model')

    # Define the dependent and independent variables
    y = df['Sales']
    x = df[['TV']]

    # Add a constant to the independent variables
    x = sm.add_constant(x)

    # Fit the model
    model = sm.OLS(y, x).fit()

    # Print the model summary
    st.subheader('Model Summary')
    st.text(model.summary())

    # Plot the data and the fitted line
    st.subheader('Scatter plot of TV and Sales with fitted line')
    fig, ax = plt.subplots()
    ax.scatter(x['TV'], y)
    ax.plot(x['TV'], model.predict(x), color='red')
    ax.set_xlabel('TV')
    ax.set_ylabel('Sales')
    ax.set_title('Scatter plot of TV and Sales with fitted line')
    st.pyplot(fig)

    # Pairplot with the fitted line
    if st.checkbox('Show pairplot with fitted line'):
        sns.pairplot(df, x_vars=['TV'], y_vars='Sales', kind='reg')
        st.pyplot(plt)
else:
    st.info('Please upload a CSV file to get started.')

