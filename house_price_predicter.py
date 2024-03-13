# These are my exercise codes to simply learn machine learning



import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split   # to break up the data in two pieces.
# We'll use some of that data as training data to fit the model, and we'll use the other data
# as validation data to calculate mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print(melbourne_data)
# print(melbourne_data.describe())
# print(melbourne_data.columns)
melbourne_data = melbourne_data.dropna(axis=0)
# dropna(axis=0) or simply dropna(): Drops rows with missing values. dropna(axis=1): Drops columns with missing values.

y = melbourne_data.Price
print(y)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X)
print(X.describe())
print(X.head())  # shows the first 5 rows of dataset

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# Same random_state: If you use the same random_state value across different runs or different instances of the model,
# you'll get the same random behavior each time. This ensures consistency in your results and makes it easier to debug
# and compare different models.
# Different random_state: If you change the random_state value,
# you'll get a different random behavior each time you run your code or initialize the model.
# This can be useful for exploring how variations in randomness affect your model's performance.

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
# print(melbourne_data.Price)          #original values
# print(melbourne_model.predict(X))    #predicted values

predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)    # val for testing part


# define the model
melbourne_model = DecisionTreeRegressor()
# fit the model
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
# print(mean_absolute_error(val_y, val_predictions))


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

# results : 500 is the optimal leaf node number

# in a shorter and more professional way
candidate_max_leaf_nodes = [5,50,500,5000]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)

# Random Forest part
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X,train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))




