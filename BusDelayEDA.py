# Author: Vijayalakshmi Gunasekarapandian

# Task 1
# Exploratory data analysis of bus breakdowns and delays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import re
from sklearn import tree
from sklearn import cross_validation

# Reading the csv file
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Bus_Breakdown_and_Delays.csv')
# print(data.info)
# print(data.describe())
# print(data.head())
# print(data.columns)

# Bar plot for Type of Incident against its count
reason_count = data['Reason'].value_counts()
plt.figure(figsize=(16, 8))
sns.barplot(x=reason_count.index, y=reason_count)
plt.ylabel('No of Occurrences')
plt.xlabel('Type of Incident')
plt.title('Incident Vs Occurrence Count')
plt.show()
# Most recurring cause of delay is heavy traffic

# Bar plot to understand the route where delays are maximum
route_count = data['Route_Number'].value_counts()
route_count_10 = route_count.sort_values().tail(10)
plt.figure(figsize=(16, 8))
sns.barplot(x=route_count_10.index, y=route_count_10)
plt.xlabel('Route Number')
plt.ylabel('Number of delay incidents')
plt.title('Delay count against Route')
plt.show()
# Route 1 faces maximum delays

# Number of Incidents in each Borough
borough_count = data['Boro'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=borough_count.index, y=borough_count)
plt.xlabel('Borough')
plt.ylabel('Number of Incidents')  #
plt.title('Number of Incidents in each Borough')
plt.show()

# Bus company with most delays
busCompany_count = data['Bus_Company_Name'].value_counts()
busCompany_count_15 = busCompany_count.sort_values().tail(15)
plt.figure(figsize=(16, 8))
sns.barplot(x=busCompany_count_15.index, y=busCompany_count_15)
plt.xlabel('Bus Company Name')
# Rotates x label  vertically
plt.xticks(rotation=90)
plt.ylabel('Number of incidents')
plt.title('Number of Incidents against Bus Company')
plt.show()

# Plotting notification
schools_notified = data['Has_Contractor_Notified_Schools'].value_counts()
plt.figure(figsize=(16, 8))
sns.barplot(x=schools_notified.index, y=schools_notified)
plt.xlabel('Schools Notified')
plt.ylabel('Count')
plt.title('Were Schools notified of the delay')
plt.show()

parents_notified = data['Has_Contractor_Notified_Parents'].value_counts()
plt.figure(figsize=(16, 8))
sns.barplot(x=parents_notified.index, y=parents_notified)
plt.xlabel('Parents Notified')
plt.ylabel('Count')
plt.title('Were Parents notified of the delay')
plt.show()

# Bar for Students delayed by academic year
delayby_year = data.groupby(['School_Year'])['Number_Of_Students_On_The_Bus'].aggregate(np.sum).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(y='Number_Of_Students_On_The_Bus', x='School_Year', data=delayby_year)
plt.title("Students delayed vs Academic Year")
plt.show()

# Pivot
grouped_year = data.groupby(["School_Year", "Reason"])["Number_Of_Students_On_The_Bus"].aggregate(np.sum).reset_index()
grouped_year = grouped_year.pivot('School_Year', 'Reason', 'Number_Of_Students_On_The_Bus')
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_year)
plt.title("Student count for reasons that cause delay each academic year")
plt.show()

# Run type that faces most delay
runtype_studentcount = data.groupby(['Run_Type'])['Number_Of_Students_On_The_Bus'].aggregate(np.sum).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(y='Number_Of_Students_On_The_Bus', x='Run_Type', data=runtype_studentcount)
plt.title("Run type that causes most students to be delayed")
plt.show()
# Special ED AM run causes most delay

# what is the most common cause of delay for Special ED AM run?
grouped_runtype = data.groupby(["Run_Type", "Reason"])["Number_Of_Students_On_The_Bus"].aggregate(np.sum).reset_index()
grouped_runtype = grouped_runtype.pivot('Run_Type', 'Reason', 'Number_Of_Students_On_The_Bus')
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_runtype)
plt.title("Most common cause of delay for a run type")
plt.show()
# Heavy traffic causes most delay during the Speical ED AM run

# which borough faces most traffic?
grouped_boro = data.groupby(["Boro", "Reason"])["Number_Of_Students_On_The_Bus"].aggregate(np.sum).reset_index()
grouped_boro = grouped_boro.pivot('Boro', 'Reason', 'Number_Of_Students_On_The_Bus')
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_boro)
plt.title("Most common cause of delay in a borough")
plt.show()
# Most students are delayed due to heavy traffic in manhattan
# Queens and Brooklyn are a close second and third

grouped_run = data.groupby(["Boro", "Run_Type"])["Number_Of_Students_On_The_Bus"].aggregate(np.sum).reset_index()
grouped_run = grouped_run.pivot('Boro', 'Run_Type', 'Number_Of_Students_On_The_Bus')
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_run)
plt.title("Most common run type in a borough")
plt.show()

