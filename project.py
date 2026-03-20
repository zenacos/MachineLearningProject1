import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('trainDATA.csv')
test_df = pd.read_csv('testDATA.csv')
train_df = train_df.dropna()
test_df = test_df.dropna()

y_train = train_df['selling_price'].values.astype(float)
y_test = test_df['selling_price'].values.astype(float)
y_mean = y_train.mean()
y_std = y_train.std()
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

owner_map = {
    'Test Drive Car': 0,
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4
}

for df in [train_df, test_df]:
    df['transmission'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})
    df['owner'] = df['owner'].map(owner_map)

features = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']

X_train_encoded = pd.get_dummies(train_df[features], columns=['fuel', 'seller_type'], drop_first=True)
X_test_encoded = pd.get_dummies(test_df[features], columns=['fuel', 'seller_type'], drop_first=True)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

num_cols = X_train_encoded.columns 
means = X_train_encoded[num_cols].mean()
stds = X_train_encoded[num_cols].std()

X_train_encoded[num_cols] = (X_train_encoded[num_cols] - means) / stds
X_test_encoded[num_cols] = (X_test_encoded[num_cols] - means) / stds

X_train_mat = X_train_encoded.values.astype(float)
X_test_mat = X_test_encoded.values.astype(float)
X_train = np.c_[np.ones(X_train_mat.shape[0]), X_train_mat]
X_test = np.c_[np.ones(X_test_mat.shape[0]), X_test_mat]

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=0.0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.theta = None
        self.cost_history = []

    def predict(self, X):
        return np.dot(X, self.theta)

    def compute_cost(self, X, y):
        m = len(y)
        predictions = self.predict(X)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def gradDescent(self, X, y):
        m = len(y)
        n_features = X.shape[1]
        self.theta = np.zeros(n_features)
        self.cost_history = []
        
        for i in range(self.max_iterations):
            predictions = self.predict(X)
            errors = predictions - y
            gradients = (1 / m) * np.dot(X.T, errors)
            self.theta -= self.learning_rate * gradients
            
            current_cost = self.compute_cost(X, y)
            self.cost_history.append(current_cost)

            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Convergence reached at iteration: {i}")
                break

def evaluate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"  MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f}")

learning_rates = [0.01, 0.05, 0.1, 0.5]
colors = ['red', 'orange', 'green', 'blue']
plt.figure(figsize=(10, 6))

for lr, color in zip(learning_rates, colors):
    print(f"\nTraining model: (Learning Rate: {lr})")
    model = LinearRegression(learning_rate=lr, max_iterations=100, tolerance=0.0)
    model.gradDescent(X_train, y_train_scaled) 
    y_pred_test_scaled = model.predict(X_test)
    y_pred_test = (y_pred_test_scaled * y_std) + y_mean
    evaluate_metrics(y_test, y_pred_test)
    plt.plot(range(len(model.cost_history)), model.cost_history, color=color, label=f'LR = {lr}')

plt.title('Effect of Different Learning Rates on Cost Function')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost (J)')
plt.legend()
plt.grid(True)
plt.show()