import numpy as np
import pandas as pd
f=pd.read_csv("Book1.csv")
print(f)

# #numpy
# #arrays
# attendance=np.array(f["Attendance"])
# print("Attendance:",attendance)

# #mean,median and mode
# marks=np.array(f["Marks"])
# print("Mean:",np.mean(marks))
# print("Median:",np.median(marks))
# # print("Mode:",np.mode(marks))

# #standard deviation and variance
# age=np.array(f["Age"])
# print("Std:",np.std(age))
# print("VAr:",np.var(marks))

# #unique values and frequenices
# unique_values= np.unique(marks)
# print("unique values", unique_values)

# #filter value using conditions
# print("Marks above 80:",marks[marks>80])
# print("marks above 50:",marks[marks<50])
# print("marks equal to 50:", marks[marks == 50])

# # Perform array arithmetic operations
# print("Addition:",marks+5)
# print("Subtraction:",marks-5)
# print("Multiplication:",marks*5)
# print("Division:",marks/5)

# # Reshape a one-dimensional array into a 2D array.
# ary=np.array([85,78,45,92,33,67,56,78,89,67])
# result=ary.reshape(5,2)
# print(result)

# # Normalize student marks between 0 and 1.
# normal=((marks-marks.min())/(marks.max()-marks.min()))
# print("normalized data",normal)

# # Calculate total sales and average units sold.
# total_marks=marks.sum()
# average_marks=marks.mean()

# print("Total marks",total_marks)
# print("Average marks",average_marks)

# # Find students scoring above 80.
# print("students scoring above 80",marks[marks>80])

# 2.Matplotlib Tasks
# Students should create charts to visualize data.
import pandas as pd
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
 
# If you haven't already loaded f, uncomment and update this line:
# f = pd.read_csv("your_file.csv")

# Clean the columns used in the charts
f["Name"] = f["Name"].fillna("Unknown").astype(str)

for column in ["Student_ID", "Marks", "Attendance", "Age"]:
    f[column] = pd.to_numeric(f[column], errors="coerce")

# Keep rows with valid values for plotting where necessary
plot_data = f.dropna(subset=["Marks", "Student_ID", "Attendance", "Age"]).copy()

# 1. Line plot: Student ID vs Marks
plt.figure()
plt.plot(plot_data["Student_ID"], plot_data["Marks"], marker="o")
plt.title("Student ID vs Marks")
plt.xlabel("Student ID")
plt.ylabel("Marks")
plt.tight_layout()
plt.show()

# 2. Bar chart: Name vs Marks
name_data = f.dropna(subset=["Marks"])
plt.figure()
plt.bar(name_data["Name"], name_data["Marks"])
plt.title("Name vs Marks")
plt.xlabel("Name")
plt.ylabel("Marks")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 3. Pie chart: students by Department
department_counts = f["Department"].fillna("Unknown").value_counts()
plt.figure()
plt.pie(
    department_counts,
    labels=department_counts.index,
    autopct="%1.1f%%"
)
plt.title("Students by Department")
plt.tight_layout()
plt.show()

# 4. Scatter plot: Attendance vs Marks
plt.figure()
plt.scatter(plot_data["Attendance"], plot_data["Marks"])
plt.title("Attendance vs Marks")
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.tight_layout()
plt.show()

# 5. Histogram: Age distribution
age_data = f["Age"].dropna()
plt.figure()
plt.hist(age_data, bins=5)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 6. Bar chart: Average Marks by Department
department_means = f.groupby("Department", dropna=False)["Marks"].mean().dropna()
plt.figure()
plt.bar(department_means.index.astype(str), department_means.values)
plt.title("Average Marks by Department")
plt.xlabel("Department")
plt.ylabel("Average Marks")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 7. Line chart: Age vs Marks
age_marks = plot_data.sort_values("Age")
plt.figure()
plt.plot(age_marks["Age"], age_marks["Marks"], marker="o")
plt.title("Age vs Marks")
plt.xlabel("Age")
plt.ylabel("Marks")
plt.tight_layout()
plt.show()

# 8. Bar chart: Name vs Attendance
attendance_data = f.dropna(subset=["Attendance"])
plt.figure()
plt.bar(attendance_data["Name"], attendance_data["Attendance"])
plt.title("Name vs Attendance")
plt.xlabel("Name")
plt.ylabel("Attendance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Seaborn charts
sns.set_theme(style="whitegrid")

# 9. Count plot: students by Department
plt.figure()
sns.countplot(x="Department", data=f, order=f["Department"].dropna().unique())
plt.title("Number of Students by Department")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 10. Box plot: Marks by Department
plt.figure()
sns.boxplot(x="Department", y="Marks", data=f)
plt.title("Marks by Department")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 11. Heatmap: correlations between numeric columns
correlation = f[["Age", "Marks", "Attendance"]].corr()
plt.figure()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 12. Pair plot: Marks and Age
sns.pairplot(f[["Marks", "Age"]].dropna())
plt.show()

# 13. Regression plot: Attendance vs Marks
regression_data = f.dropna(subset=["Attendance", "Marks"])
sns.regplot(x="Attendance", y="Marks", data=regression_data)
plt.title("Attendance vs Marks Regression")
plt.tight_layout()
plt.show()

# 14. Scatter plot: Age vs Attendance
scatter_data = f.dropna(subset=["Age", "Attendance"])
sns.scatterplot(x="Age", y="Attendance", data=scatter_data)
plt.title("Age vs Attendance")
plt.tight_layout()
plt.show()

# 15. Histogram: Marks distribution
plt.figure()
sns.histplot(x="Marks", bins=5, data=f)
plt.title("Marks Distribution")
plt.xlabel("Marks")
plt.tight_layout()
plt.show()

# 16. Box plot: Attendance distribution
plt.figure()
sns.boxplot(x="Attendance", data=f)
plt.title("Attendance Distribution")
plt.tight_layout()
plt.show()
