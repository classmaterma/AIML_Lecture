import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Prediction")
st.write("predicts Iris flower species based on the input parameters.")

iris = load_iris()
x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]
print(df)

st.sidebar.header("Input Parameters")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length (cm)", 
                                        float(df['sepal length (cm)'].min()),
                                        float(df['sepal length (cm)'].max()),
                                        float(df['sepal length (cm)'].mean())
                                     )
    sepal_width = st.sidebar.slider("Sepal width (cm)", 
                                        float(df['sepal width (cm)'].min()),
                                        float(df['sepal width (cm)'].max()),
                                        float(df['sepal width (cm)'].mean())
                                     )
    petal_length = st.sidebar.slider("Petal length (cm)", 
                                        float(df['petal length (cm)'].min()),
                                        float(df['petal length (cm)'].max()),
                                        float(df['petal length (cm)'].mean())
                                     )
    petal_width = st.sidebar.slider("Petal width (cm)", 
                                        float(df['petal width (cm)'].min()),
                                        float(df['petal width (cm)'].max()),
                                        float(df['petal width (cm)'].mean())
                                     )
    data = {'Sepal length (cm)': sepal_length,
            'Sepal width (cm)': sepal_width,
            'Petal length (cm)': petal_length,
            'Petal width (cm)': petal_width
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.subheader("User input Parameters")
st.write(input_df)

model = RandomForestClassifier()
model.fit(x,y)

predict = model.predict(input_df.to_numpy())
predict_prob = model.predict_proba(input_df.to_numpy())

st.subheader("Prediction")
st.write(iris.target_names[predict])

st.subheader("Prediction Probability")
st.write(predict_prob)

st.subheader("Feature Importance")
importances = model.feature_importances_
print(importances)
indices = np.argsort(importances)[::-1]
print(indices)
plt.figure( figsize = (10,4))
plt.title("Feature Importance")
plt.bar(range(x.shape[1]) , importances[indices])
plt.xticks(range(x.shape[1]), [iris.feature_names[i] for i in indices])
print(iris.feature_names)
plt.xlim([-1, x.shape[1]])
st.pyplot(plt)

st.subheader("Histogram of Features")
fig, axes = plt.subplots(2,2, figsize= (12,8))
axes = axes.flatten()
for i, ax in enumerate(axes):
   sns.histplot(df[iris.feature_names[i]])
   ax.set_title(iris.feature_names[i])
plt.tight_layout()
st.pyplot(fig)


plt.figure(figsize = (10,8))
st.subheader("Correction Matrix")
numerical_df = df.drop("species", axis = 1)
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", fmt = '.2f', linewidth = 0.5)
plt.tight_layout()
st.pyplot(plt)

st.subheader("Pairplot")
fig = sns.pairplot(df, hue = "species")
plt.tight_layout()
st.pyplot(fig)
