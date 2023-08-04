
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# Load data from CSV file
#with open('C:/Users/jalaj/VsCodeLiter/PYs/sales_prediction_project data/Advertising.csv') as file:
 #   data = csv.reader(file)

data = pd.read_csv('C:/Users/jalaj/VsCodeLiter/PYs/sales_prediction_project data/Advertising.csv')

    

# Define function to perform linear regression analysis
def perform_linear_regression(data):
    # Extract features and target variables
    X = data[['Radio', 'TV', 'Newspaper']]
    y = data['Sales']
    
    # Fit linear regression model
    model = LinearRegression().fit(X, y)
    
    # Calculate R-squared score
    r2 = model.score(X, y)
    
    # Return results
    return {'model': 'Linear Regression', 'R-squared': r2}

# Define function to perform regularization analysis
def perform_regularization(data, alpha, regularization):

    model = None
    # Extract features and target variables
    X = data[['Radio', 'TV', 'Newspaper']]
    y = data['Sales']
    
    # Choose regularization type
    if regularization == 'Ridge':
        model = Ridge(alpha=alpha).fit(X, y)
    elif regularization == 'Lasso':
        model = Lasso(alpha=alpha).fit(X, y)
    
    # Check if model object exists
    if model is not None:
        # Calculate R-squared score
        r2 = model.score(X, y)
        
        # Return results
        return {'model': regularization, 'alpha': alpha, 'R-squared': r2}
    else:
        # Handle case where model object does not exist
        print('Error: Could not fit model with given dataset and parameters.')
        return None


# Define function to perform descriptive analysis
def perform_descriptive_analysis(data):
    # Calculate descriptive statistics
    desc_stats = data.describe()
    
    # Create histograms for each feature
    data.hist()
    plt.show()
    
    # Create scatterplots for each pair of features
    sns.pairplot(data)
    plt.show()
    
    # Return results
    return {'stats': desc_stats, 'plots': 'Histograms and Scatterplots'}

# Define function to perform analysis based on user input
def perform_analysis(data, option):
    if option == 1:
        return perform_linear_regression(data)
    elif option == 2:
        alpha = float(input('Enter regularization parameter alpha: '))
        regularization = input('Enter regularization type (Ridge or Lasso): ')
        return perform_regularization(data, alpha, regularization)
    elif option == 3:
        return perform_descriptive_analysis(data)
    else:
        print('Invalid option.')

# Define function to get user input for analysis option
def get_user_option():
    print('Select an analysis option:')
    print('1. Linear Regression')
    print('2. Regularization')
    print('3. Descriptive Analysis')
    option = int(input('Enter option number: '))
    return option

# Get user input and perform analysis
option = get_user_option()
result = perform_analysis(data, option)
print(result)


#Lets check out option 3 now.