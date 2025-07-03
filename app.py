import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 设置页面配置
st.set_page_config(
    page_title="Surgery Risk Prediction",
    page_icon="⚕️",
    layout="wide"
)

# 设置中文字体（确保SHAP图正确显示中文）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 标题和介绍
st.title("Unplanned Reoperation Risk Prediction")
st.markdown("This application uses machine learning to predict the risk of unplanned reoperation based on patient characteristics.")

# 加载数据函数（使用示例数据，实际应用中替换为您的数据）
@st.cache_data
def load_data():
    # 创建示例数据（实际应用中替换为pd.read_excel('your_data.xlsx')）
    data = {
        'Sex': np.random.randint(0, 2, 100),
        'ASA scores': np.random.randint(1, 5, 100),
        'tumor location': np.random.randint(1, 6, 100),
        'Benign or malignant': np.random.randint(0, 2, 100),
        'Admitted to NICU': np.random.randint(0, 2, 100),
        'Duration of surgery': np.random.randint(0, 3, 100),
        'diabetes': np.random.randint(0, 2, 100),
        'CHF': np.random.randint(0, 2, 100),
        'Functional dependencies': np.random.randint(0, 2, 100),
        'mFI-5': np.random.randint(1, 6, 100),
        'Type of tumor': np.random.randint(1, 6, 100),
        'Unplanned reoperation': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)
    return df

# 训练模型函数
@st.cache_data
def train_model(df):
    # 提取特征和目标变量
    X = df.drop('Unplanned reoperation', axis=1)
    y = df['Unplanned reoperation']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练随机森林模型（可替换为其他模型）
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns

# 加载数据和训练模型
df = load_data()
model, feature_names = train_model(df)

# 用户输入表单
st.sidebar.header("Patient Characteristics")
st.sidebar.markdown("Adjust the parameters below to predict the risk of unplanned reoperation.")

# 创建输入特征
input_features = {}
for feature in feature_names:
    min_val = int(df[feature].min())
    max_val = int(df[feature].max())
    default_val = int(df[feature].mean())
    
    # 根据特征类型选择合适的输入控件
    if len(df[feature].unique()) <= 3:  # 分类特征
        input_features[feature] = st.sidebar.select_slider(
            feature,
            options=list(range(min_val, max_val + 1)),
            value=default_val
        )
    else:  # 连续特征
        input_features[feature] = st.sidebar.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=default_val
        )

# 预测按钮
if st.sidebar.button("Predict Risk"):
    # 创建预测输入
    input_df = pd.DataFrame([input_features])
    
    # 预测风险
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    
    # 显示预测结果
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"**Prediction:** High Risk of Unplanned Reoperation")
        st.error(f"**Probability:** {probability:.2f}%")
    else:
        st.success(f"**Prediction:** Low Risk of Unplanned Reoperation")
        st.success(f"**Probability:** {probability:.2f}%")
    
    # 使用SHAP解释预测
    st.subheader("Prediction Explanation")
    st.markdown("The following SHAP plot shows how each feature influenced the prediction:")
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # 绘制SHAP解释图
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][0],
        input_df.iloc[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    
    # 显示特征重要性排序
    st.subheader("Feature Importance")
    feature_importance = pd.Series(
        np.abs(shap_values[1][0]),
        index=feature_names
    ).sort_values(ascending=False)
    
    st.dataframe(feature_importance.reset_index().rename(
        columns={"index": "Feature", 0: "SHAP Value (Impact)"}
    ))    