import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


st.set_page_config(
    page_title="Breast Cancer Detection Dashboard",
    page_icon="🩺",
    layout="wide"
)


@st.cache_data
def load_data():
    data = load_breast_cancer()

    df = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    df["target"] = data.target
    df["diagnosis"] = df["target"].map(
        {0: "Malignant", 1: "Benign"}
    )

    return df

df = load_data()


st.sidebar.title("Filters")

diagnosis_filter = st.sidebar.multiselect(
    "Select Diagnosis",
    options=df["diagnosis"].unique(),
    default=df["diagnosis"].unique()
)

filtered_df = df[df["diagnosis"].isin(diagnosis_filter)]


st.title("🩺 Breast Cancer Detection Dashboard")

st.markdown("""
This dashboard provides exploratory analysis and machine learning based prediction
using the Breast Cancer Wisconsin Diagnostic Dataset.
""")


st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Features", len(df.columns)-2)
col3.metric("Filtered Records", len(filtered_df))

st.dataframe(filtered_df.head())

st.header("Diagnosis Distribution")

fig = px.pie(
    filtered_df,
    names="diagnosis",
    title="Benign vs Malignant Cases"
)

st.plotly_chart(fig, use_container_width=True)


st.header("Feature Analysis")

feature = st.selectbox(
    "Select Feature",
    df.columns[:-2]
)

hist = px.histogram(
    filtered_df,
    x=feature,
    color="diagnosis",
    barmode="overlay",
    title=f"Distribution of {feature}"
)

st.plotly_chart(hist, use_container_width=True)


st.header("Feature Relationship")

col1, col2 = st.columns(2)

x_feature = col1.selectbox(
    "X Axis",
    df.columns[:-2],
    index=0
)

y_feature = col2.selectbox(
    "Y Axis",
    df.columns[:-2],
    index=1
)

scatter = px.scatter(
    filtered_df,
    x=x_feature,
    y=y_feature,
    color="diagnosis",
    title=f"{x_feature} vs {y_feature}"
)

st.plotly_chart(scatter, use_container_width=True)


st.header("Correlation Heatmap")

corr = df.drop(
    columns=["target", "diagnosis"]
).corr()

heatmap = px.imshow(
    corr,
    aspect="auto",
    title="Feature Correlation Matrix"
)

st.plotly_chart(heatmap, use_container_width=True)


st.header("Machine Learning Model")

X = df.drop(columns=["target", "diagnosis"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.success(
    f"Model Accuracy: {accuracy*100:.2f}%"
)


cm = confusion_matrix(
    y_test,
    y_pred
)

fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=["Malignant", "Benign"],
    y=["Malignant", "Benign"],
    showscale=True
)

fig_cm.update_layout(
    title="Confusion Matrix"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)


st.header("Predict Tumor Type")

st.write(
    "Enter sample values below:"
)

sample_input = {}

for feature in X.columns[:10]:
    sample_input[feature] = st.number_input(
        feature,
        value=float(df[feature].mean())
    )

if st.button("Predict"):

    input_df = pd.DataFrame(
        [sample_input]
    )

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = df[col].mean()

    input_df = input_df[X.columns]

    prediction = model.predict(
        input_df
    )[0]

    if prediction == 0:
        st.error(
            "Prediction: Malignant"
        )
    else:
        st.success(
            "Prediction: Benign"
        )


st.markdown("---")
st.markdown(
    "Developed using Python, Streamlit, Plotly and Scikit-Learn"
)