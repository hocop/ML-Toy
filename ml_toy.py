import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datasets import dataset_functions

metrics = {
    'Classification': [
        ('f1 score', f1_score),
        ('accuracy', accuracy_score),
        ('precision', precision_score),
        ('recall', recall_score),
    ],
    'Regression': [
        ('MSE', mean_squared_error),
        ('RMSE', lambda true, pred: np.sqrt(mean_squared_error(true, pred))),
        ('MAE', mean_absolute_error),
        ('R2', r2_score),
    ]
}

st.title('ML Toy')
st.write('''
Here you can play with very simple machine learning tasks. I hope it will help you understand better how machine learning models work.
''')

# Select task
st.header('Dataset')
task_sb = st.selectbox(
    'Task',
    ['Classification', 'Regression']#, 'Clustering']
)
function_sb = st.selectbox(
    'Dataset',
    list(dataset_functions[task_sb])
)
df_function = dataset_functions[task_sb][function_sb]

# Configure dataset
st.sidebar.header('Configure dataset')
num_samples = st.sidebar.slider('Number of samples', 10, 1000, 100)
noise = st.sidebar.slider('Amount of noise in training set', 0.0, 1.0, 0.0)
coef = st.slider('K coefficient', 0.0, 10.0, 2.0)
feature0 = st.sidebar.text_input('Name of feature 0', 'Age')
if task_sb == 'Classification':
    feature1 = st.sidebar.text_input('Name of feature 1', 'Income')
elif task_sb == 'Regression':
    feature1 = st.sidebar.text_input('Name of target variable', 'Income')

@st.cache
def get_dataset(num_samples, coef, noise, task_sb):
    if task_sb == 'Classification':
        X = np.random.normal(size=[num_samples, 2])
        y = df_function(X, coef)
        mask = np.random.random(size=y.shape) < noise
        y[mask] = np.random.random(size=y[mask].shape) < 0.5
    elif task_sb == 'Regression':
        X = np.random.normal(size=[num_samples, 1])
        y = df_function(X, coef)
        mask = np.random.random(size=y.shape) < noise
        y[mask] = np.random.normal(size=y[mask].shape)
    return X, y

X_train, y_train = get_dataset(num_samples, coef, noise, task_sb)
X_test, y_test = get_dataset(1000, coef, 0, task_sb)

# Plot dataset
@st.cache
def plot_dataset(axes, alpha=0.5):
    axes[0].set_title('Train set')
    axes[1].set_title('Test set')
    if task_sb == 'Classification':
        axes[0].scatter(X_train[y_train == 0, 0],
                        X_train[y_train == 0, 1],
                        label='class 0',
                        alpha=alpha)
        axes[0].scatter(X_train[y_train == 1, 0],
                        X_train[y_train == 1, 1],
                        label='class 1',
                        alpha=alpha,
                        color='red')
        axes[1].scatter(X_test[y_test == 0, 0],
                        X_test[y_test == 0, 1],
                        label='class 0',
                        alpha=alpha)
        axes[1].scatter(X_test[y_test == 1, 0],
                        X_test[y_test == 1, 1],
                        label='class 1',
                        alpha=alpha,
                        color='red')
    elif task_sb == 'Regression':
        axes[0].scatter(X_train[:, 0],
                        y_train,
                        label='Train',
                        alpha=alpha)
        axes[1].scatter(X_test[:, 0],
                        y_test,
                        label='Test',
                        alpha=alpha)
    for ax in axes:
        ax.set_aspect(1)
        ax.legend()
        ax.set_xlabel(feature0)
        ax.set_ylabel(feature1)
        ax.set_xlim((-3, 3))
        ax.set_ylim((-2, 2))

fig, axes = plt.subplots(1, 2, figsize=(10,5))
plot_dataset(axes)
st.pyplot()


# Select model
st.sidebar.header('Configure model')
lines = open('models.json').read()
models = json.loads(lines)
model_sb = st.sidebar.selectbox(
    'Select a model',
    [name for name in models if models[name]['type'] == task_sb]
)

# Get model class name
model = models[model_sb]
import_what = model['import'].split('.')[-1]
import_from = '.'.join(model['import'].split('.')[:-1])

# Select model parameters
params = {}
for key in model['params']:
    values = model['params'][key]
    if isinstance(values, list):
        value = st.sidebar.selectbox(key, values)
    elif isinstance(values, dict):
        value = st.sidebar.slider(key, values['from'], values['to'], values['default'])
    params[key] = value

# Model creation code
model_code = '''
from {} import {}

params = {}
model = {}(**params)
'''.format(import_from,
           import_what,
           str(params),
           import_what,
    )

st.header('Model')
st.write('Configure your model on the left sidebar')
st.write('Your code:')
st.write('```python\n' + model_code + '\n```')

# Train the model
exec(model_code)
model.fit(X_train, y_train)

# Plot predictions
fig, axes = plt.subplots(1, 2, figsize=(10,5))
if task_sb == 'Classification':
    # Heatmap
    plot_dataset(axes)
    X,Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    X_map = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)
    if task_sb == 'Classification' and st.checkbox('Show probability', False):
        try:
            Z = model.predict_proba(X_map)[:,1].reshape(X.shape[:2])
        except:
            Z = model.predict(X_map).reshape(X.shape[:2])
    else:
        Z = model.predict(X_map).reshape(X.shape[:2])
    axes[0].pcolormesh(X,Y,Z, cmap='coolwarm', alpha=0.5)
    axes[1].pcolormesh(X,Y,Z, cmap='coolwarm', alpha=0.5)
elif task_sb == 'Regression':
    # Plot
    X = np.linspace(X_test.min(), X_test.max(), 1000).reshape([-1, 1])
    y = model.predict(X)
    axes[0].plot(X, y, alpha=0.8, linewidth=2, color='red', label=import_what)
    axes[1].plot(X, y, alpha=0.8, linewidth=2, color='red', label=import_what)
    plot_dataset(axes)
st.pyplot()

# Print metrics
predictions = model.predict(X_test)
results = {}
for metric_name, metric in metrics[task_sb]:
    results[metric_name] = metric(y_test, predictions)
st.write('Results on test set:')
st.write(pd.Series(results))

# Plot tree
if 'Decision Tree' in model_sb and st.checkbox('Draw tree', False):
    from sklearn import tree
    tree.plot_tree(model,
                   feature_names=[feature0, feature1],
                   filled=True,
                   label='root',
    )
    st.pyplot()

st.write('---')
st.button('reload')

st.write('''
---
P.S. This app is completely open-source: https://github.com/hocop/ML-Toy
''')




