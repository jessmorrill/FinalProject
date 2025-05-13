import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind, f_oneway
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title="Mental Health & Social Media Dashboard", layout="wide")
st.title("📱 Mental Health and Social Media Usage Dashboard")
st.write("First we will look at the correlations in the first dataset")

@st.cache_data
def load_data():
    url_social = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_1SGG51k-OJbCzzFck7SFQOSt4EvafRqwedPaxyyIzIrie_RdcuZcfOU9SYu4AQImcMJFEVNqO-Ma/pub?output=csv"
    url_platform = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQzRy338yxDiqPrifzNmXLYdh6toGhvRoArsA3vMd4Cwt5LrfhvzmQULVL6KYYqSFFdCVWlasoEuA15/pub?output=csv"
    return pd.read_csv(url_social), pd.read_csv(url_platform)

df_social, df_platform = load_data()

mh_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
df_social['Mental_Health_Score'] = df_social['Mental_Health_Status'].map(mh_map)
df_social['Support_Systems_Access'] = df_social['Support_Systems_Access'].astype(str)
df_social = df_social.dropna(subset=['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours', 'Mental_Health_Score'])

support_filter = st.selectbox("Support System Access", ["All", "Yes", "No"])
if support_filter != "All":
    df_social = df_social[df_social['Support_Systems_Access'] == support_filter]

st.subheader("Heatmap - Correlation Between Sleep, Activity, Screen Time, and Mental Health")
corr_vars = ['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours', 'Mental_Health_Score']
corr = df_social[corr_vars].corr()
fig1, ax1 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
st.pyplot(fig1)

st.subheader("Regression Model, Sleep, Activity, Screen Time vs Mental Health")
X_sm = df_social[['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours']]
X_sm = sm.add_constant(X_sm)
y_sm = df_social['Mental_Health_Score']
model_sm = sm.OLS(y_sm, X_sm).fit()
st.text(model_sm.summary())

st.subheader("Linear Regresssion using scikit learn")
X = df_social[['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours']]
y = df_social['Mental_Health_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"RMSE: {rmse:.4f}")

st.subheader("Another Model - Logistic Regression")
df_social['MH_Class'] = (df_social['Mental_Health_Score'] >= 3).astype(int)
X_class = df_social[['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours']]
y_class = df_social['MH_Class']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

st.write("### Feature Coefficients")
coef_df = pd.DataFrame({
    'Feature': X_class.columns,
    'Coefficient': clf.coef_[0]
})
st.bar_chart(coef_df.set_index('Feature'))

st.title("Does the type of screen someone is using impact their mental health?")

#Plot One from Slideshow
st.header = "Average Sreentime Makeup by Mental Health Status"
avg_usage = df_social.groupby('Mental_Health_Status')[['Technology_Usage_Hours', 'Social_Media_Usage_Hours', 'Gaming_Hours']].mean()
avg_usage = avg_usage.reindex(['Poor', 'Fair', 'Good', 'Excellent'])
total_screen_time = avg_usage.sum(axis=1)
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(avg_usage.index, avg_usage['Technology_Usage_Hours'], label='Technology Usage', color='skyblue', edgecolor="black")
ax.bar(avg_usage.index, avg_usage['Social_Media_Usage_Hours'], 
       bottom=avg_usage['Technology_Usage_Hours'], label='Social Media Usage', color='#FF9999', edgecolor = "black")
ax.bar(avg_usage.index, avg_usage['Gaming_Hours'], 
       bottom=avg_usage['Technology_Usage_Hours'] + avg_usage['Social_Media_Usage_Hours'], 
       label='Gaming', color='lightgreen', edgecolor = "black")


for i, total in enumerate(total_screen_time):
    ax.text(i, total + 0.2, f'{total:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Mental Health Status')
ax.set_ylabel('Average Hours')
ax.set_title('Average Screen Time Makeup by Mental Health Status')
ax.legend()
st.pyplot(fig)
st.write("As seen in this chart, there is very little difference in screen time makeup across all mental health statuses. This is surprising and contradicts our original prediction. Most people achieve a static balance between technology, gaming, and social media.")

##Plot Three From Our Slideshow
st.title("How does someone's age and gender impact their screen time hours?")
age_bins = [17, 25, 35, 45, 55, 65]
age_labels = ['18–25', '26–35', '36–45', '46–55', '56–65']
df_social['age_group'] = pd.cut(df_social['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

col1, col2 = st.columns(2)

with col1:
    gender_filter = st.radio(
        "Select Gender",
        options=["All", "Male", "Female"],
        index=0
    )

with col2:
    age_filter = st.radio(
        "Select Age Group",
        options=["All"] + age_labels,
        index=0
    )

filtered_df = df_social
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
if age_filter != "All":
    filtered_df = filtered_df[filtered_df['age_group'] == age_filter]

fig, ax = plt.subplots(figsize=(12, 6))
    
ax.hist(filtered_df['Screen_Time_Hours'], bins=20, color='skyblue', edgecolor='black')
    
ax.set_xlabel('Total Screen Time (Hours)')
ax.set_ylabel('Frequency')
ax.set_title(f'Distribution of Total Screen Time\n(Gender: {gender_filter}, Age Group: {age_filter})')
    
st.pyplot(fig)
st.write("This chart shows how often people are on screens by their age. This is meant to address the hypothesis that younger age groups are on screens longer.")
st.title("Social Media Platform Usage vs. Depression Levels")

df = pd.read_csv("smmh.csv")

if df is not None:
    # Filter for users who use social media
    df = df[df["6. Do you use social media?"] == "Yes"].copy()

    depression_col = "18. How often do you feel depressed or down?"
    df["Mental_Health_Status"] = df[depression_col]

    platforms_col = "7. What social media platforms do you commonly use?"
    unique_platforms = (
        df[platforms_col]
        .str.split(", ")
        .explode()
        .str.strip()
        .unique()
    )
    for platform in unique_platforms:
        df[platform] = df[platforms_col].apply(
            lambda x: 1 if platform in str(x).split(", ") else 0
        )

    st.subheader("Bar Plot of Users by Depression Level")
    platform_options = ["All"] + list(unique_platforms)
    selected_platform = st.selectbox("Select a Social Media Platform:", platform_options)

    if selected_platform == "All":
        plot_data = df["Mental_Health_Status"].value_counts().reindex(range(1,6), fill_value=0)
        title = "Number of Social Media Users by Depression Level"
    else:
        plot_data = df[df[selected_platform] == 1]["Mental_Health_Status"].value_counts().reindex(
            range(1,6), fill_value=0
        )
        title = f"Number of {selected_platform} Users by Depression Level"

    plot_df = pd.DataFrame({
        "Depression_Level": [i for i in range (1,6)],
        "Count": [plot_data.get(i, 0) for i in range(1,6)]
    })

    fig = px.bar(
        plot_df,
        x="Depression_Level",
        y="Count",
        title=title,
        labels={"Count": "Number of Users", "Depression_Level": "Responses to the Question \"On a scale of 1-5, How often do you feel depressed or down?\""},
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig)
    st.write("This chart looks at what social media are being reported by those who feel depressed. Snapchat and Tiktok stand out as having high numbers of users who report feeling depressed.")
time_mapping = {
    'Less than an Hour': 0.5,
    'Between 2 and 3 hours': 2.5,
    'Between 3 and 4 hours': 3.5,
    'More than 5 hours': 5.5
}

df['Social Media Time (Hours)'] = df['8. What is the average time you spend on social media every day?'].map(time_mapping)
# Filter out rows with NaN in 'Social Media Time (Hours)'
df = df.dropna(subset=['Social Media Time (Hours)'])

# Create age groups
def categorize_age(age):
    if age < 20:
        return '< 20'
    elif 20 <= age < 30:
        return '20-29'
    elif 30 <= age < 40:
        return '30-39'
    elif 40 <= age < 50:
        return '40-49'
    elif 50 <= age < 60:
        return '50-59'
    else:
        return '60+'


df['Age Group'] = df['1. What is your age?'].apply(categorize_age)
df = df.dropna(subset=['Age Group'])  # Remove rows where age couldn't be categorized


st.title("Does more time on social media lead to trouble sleeping?")

def sort_age_groups(age_group):
    if age_group == '< 20':
        return 0
    elif age_group == '60+':
        return 60
    else:
        return int(age_group.split('-')[0])

unique_age_groups = sorted(df['Age Group'].unique(), key=sort_age_groups)
selected_age_group = st.radio("Select Age Group", ['All'] + unique_age_groups, index=0)

if selected_age_group == 'All':
    filtered_df = df.copy() 
else:
    filtered_df = df[df['Age Group'] == selected_age_group]  # Filter by age group
#groupby
agg_df = filtered_df.groupby(['Social Media Time (Hours)', '8. What is the average time you spend on social media every day?'])['20. On a scale of 1 to 5, how often do you face issues regarding sleep?'].mean().reset_index()
agg_df = agg_df.sort_values('Social Media Time (Hours)')  

# line chart add
fig = px.line(
    agg_df,
    x='Social Media Time (Hours)',
    y='20. On a scale of 1 to 5, how often do you face issues regarding sleep?',
    markers=True,
    hover_data=['8. What is the average time you spend on social media every day?'],
    title=f'Average Trouble Sleeping vs. Social Media Time (Age Group {selected_age_group})',
    labels={
        'Social Media Time (Hours)': 'Average Time on Social Media (Hours)',
        '20. On a scale of 1 to 5, how often do you face issues regarding sleep?': 'Average Sleep Issues (1-5 Scale)'
    }
)


fig.update_layout(
    xaxis_title="Average Time on Social Media (Hours)",
    yaxis_title="Average Trouble Sleeping (1-5 Scale)",
    showlegend=False
)
st.plotly_chart(fig)
st.write("This chart explores if high social media usage leads to sleep issues across all ages. As seen, all ages show a clear upward trend in reported issues sleeping as they spend more time on social media.")

df = df.dropna(subset=['2. Gender', '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?'])


def standardize_gender(gender):
    if gender in ['Male']:
        return 'Male'
    elif gender in ['Female']:
        return 'Female'
    else:
        return 'Other'

df['2. Gender'] = df['2. Gender'].apply(standardize_gender)

st.title("How does age and gender impact how often people compare themselves to others on social media?")


def sort_age_groups(age_group):
    if age_group == '< 20':
        return 0
    elif age_group == '60+':
        return 60
    else:
        return int(age_group.split('-')[0])


unique_age_groups = sorted(df['Age Group'].unique(), key=sort_age_groups)
selected_age_group = st.radio("Select Age Group", ['All Ages'] + unique_age_groups, index=0)


if selected_age_group == 'All Ages':
    filtered_df = df.copy()
else:
    filtered_df = df[df['Age Group'] == selected_age_group]


comparison_col = '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?'


score_dist = filtered_df.groupby(['2. Gender', comparison_col]).size().unstack(fill_value=0)
score_dist_pct = score_dist.div(score_dist.sum(axis=1), axis=0) * 100


st.subheader(f"Responses to the question: On a scale of 1-5, how often do you compare yourself to others through the use of social media?")


fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#FF9999', '#FFB366', '#FFFF66', '#B3FF66', '#66FF66']  # Light red to light green
score_labels = ['Score 1\n(Never)', 'Score 2\n(Rarely)', 'Score 3\n(Sometimes)', 
                'Score 4\n(Often)', 'Score 5\n(Always)']

genders = score_dist_pct.index
x_positions = np.arange(len(genders))


bottom_values = np.zeros(len(genders))
for i, score in enumerate(score_dist_pct.columns):
    values = score_dist_pct[score].values
    ax.bar(x_positions, values, bottom=bottom_values, 
           label=score_labels[i], color=colors[i], alpha=0.8)
    bottom_values += values


ax.set_xlabel('Gender', fontsize=12)
ax.set_ylabel('Percentage of Respondents', fontsize=12)
ax.set_title(f'Self-Comparison Score Distribution by Gender ({selected_age_group})', fontsize=14)
ax.set_xticks(x_positions)
ax.set_xticklabels(genders, fontsize=11)
ax.legend(title='Comparison Frequency', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim(0, 100)


for i, gender in enumerate(genders):
    cumulative = 0
    for j, score in enumerate(score_dist_pct.columns):
        value = score_dist_pct.loc[gender, score]
        if value > 5:  # Only show labels for segments > 5%
            ax.text(i, cumulative + value/2, f'{value:.1f}%', 
                   ha='center', va='center', fontweight='bold', fontsize=9)
        cumulative += value

plt.tight_layout()
st.pyplot(fig)
"This final bar chart looks at how often different genders and ages are using social media to compare themselves to other. As can be seen, younger ages are much more frequently comparing themselves to more successful people online."
