import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split  # Added this line
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64


app = Flask(__name__)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'GET':
        return render_template('classification.html')
    elif request.method == 'POST':
        # Get selected features and classifier
        selected_features = request.form.getlist('features')
        classifier = request.form['classifier']

        if not selected_features:
            return "Please select at least one feature.", 400

        # Prepare data
        features = ['Temperature', 'Humidity', 'PM10', 'NO2', 'SO2', 'PM2.5', 'CO',
                    'Proximity_to_Industrial_Areas', 'Population_Density']
        target = 'Air Quality'
        le = LabelEncoder()
        dataset['Air Quality'] = le.fit_transform(dataset[target])

        X = dataset[selected_features]
        y = dataset['Air Quality']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

        # Select classifier
        if classifier == 'logistic_regression':
            model = LogisticRegression(multi_class='ovr', max_iter=500)
        elif classifier == 'svm':
            model = SVC(probability=True)
        elif classifier == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        elif classifier == 'naive_bayes':
            model = GaussianNB()
        else:
            return "Invalid classifier selected.", 400

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_names = [str(cls) for cls in le.classes_]
        class_report = classification_report(y_test, y_pred, target_names=class_names)

        # ROC Curve
        plt.figure(figsize=(10, 8))
        for i in range(len(le.classes_)):
            fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {le.classes_[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Air Quality Classification')
        plt.legend(loc='lower right')

        # Save plot
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf8')
        buffer.close()
        plt.close()

        return render_template('classification.html', plot_url=plot_data, accuracy=accuracy, class_report=class_report)

# Load dataset
dataset = pd.read_csv('updated_pollution_dataset.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    if request.method == 'GET':
        return render_template('regression.html')
    elif request.method == 'POST':
        # Get user inputs
        method = request.form['method']
        target = request.form['target']
        x_axis = request.form['x_axis']
        y_axis = request.form['y_axis']

        # Define features (excluding the target feature dynamically)
        features = ['Temperature', 'Humidity', 'PM10', 'NO2', 'SO2', 'PM2.5', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
        training_features = [feature for feature in features if feature != target]  # Exclude the target
        X = dataset[training_features]
        y = dataset[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select regression model
        if method == 'linear':
            model = LinearRegression()
        elif method == 'ridge':
            model = Ridge()
        elif method == 'lasso':
            model = Lasso()
        elif method == 'decision_tree':
            model = DecisionTreeRegressor()
        elif method == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        else:
            return "Invalid regression method selected."

        # Train the model
        model.fit(X_train, y_train)

        # Generate predictions for grid
        x_values = np.linspace(X[x_axis].min(), X[x_axis].max(), 100)
        y_values = np.linspace(X[y_axis].min(), X[y_axis].max(), 100)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        # Prepare grid data for prediction
        grid_data = pd.DataFrame({
            feature: np.full(x_grid.ravel().shape, X[feature].mean()) for feature in training_features
        })
        grid_data[x_axis] = x_grid.ravel()
        grid_data[y_axis] = y_grid.ravel()

        # Predict target for grid
        predictions = model.predict(grid_data)
        prediction_grid = predictions.reshape(x_grid.shape)

        # Plot regression surface
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(x_grid, y_grid, prediction_grid, levels=20, cmap='viridis')
        plt.colorbar(contour, label=f'Predicted {target}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'Regression Surface: Predicted {target}')

        # Save plot to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf8')
        buffer.close()
        plt.close()

        return render_template('regression.html', plot_url=plot_data)


@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    if request.method == 'GET':
        return render_template('clustering.html')
    elif request.method == 'POST':
        # Get inputs from form
        k_value = int(request.form['k_value'])
        clustering_feature = request.form['feature']
        x_axis = request.form['x_axis']
        y_axis = request.form['y_axis']

        # Define all features
        all_features = ['Temperature', 'Humidity', 'PM10', 'NO2', 'SO2', 'PM2.5', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']

        # Exclude the selected clustering feature from the training features
        training_features = [feature for feature in all_features if feature != clustering_feature]

        # Prepare the data
        X = dataset[training_features]

        # Normalize the features
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k_value, random_state=42)
        dataset['Cluster'] = kmeans.fit_predict(X_normalized)

        # Prepare grid for plotting (using x_axis and y_axis)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            dataset[x_axis], dataset[y_axis], c=dataset['Cluster'], cmap='viridis'
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'Clustering Analysis (k={k_value})')

        # Save plot to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf8')
        buffer.close()
        plt.close()

        return render_template('clustering.html', plot_url=plot_data)


if __name__ == '__main__':
    app.run(debug=True)