import pandas as pd
c=pd.read_csv('Book1.csv')
print(c)
#part A
print(c.head())

print(c.tail())

print(c.describe())

print(c.info())

print(c.columns)

print(c.shape)

#part B
print(c.isnull().sum())

c['Age']=c['Age'].fillna(c['Age'].median())


c['Marks']=c['Marks'].fillna(c['Marks'].mean(),inplace=True)

print(c.isnull().sum())

print(c.duplicated())

print(c.duplicated().sum())
#part c

more=c[c['Marks']>75]
print(more)

below=c[c['Attendance']<75]
print(below)

cse=c[c['Department']=='CSE']
print(cse)

between=c[(c['Marks']>=50) & (c['Marks']<=80)]
print(between)

sorted_c=c.sort_values(by='Marks', ascending=False)
print(sorted_c)
# part d
c['Result'] = c['Marks'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')



def assign_grade(marks):
    if marks >= 90:
        return 'A+'
    elif marks >= 80:
        return 'A'
    elif marks >= 70:
        return 'B'
    elif marks >= 60:
        return 'C'
    else:
        return 'D'

c['Grade'] = c['Marks'].apply(assign_grade)

c = c.rename(columns={'Marks': 'Total_Marks'})

c['Attendance'] = pd.to_numeric(c['Attendance'], errors='coerce')

c = c.reset_index(drop=True)
# part e
average_marks = c['Total_Marks'].mean()
print("Average Marks of all students:", average_marks)

dept_avg_marks = c.groupby('Department')['Total_Marks'].mean()
print("Average Marks by Department:")
print(dept_avg_marks)

dept_student_count = c.groupby('Department').size()
print("Number of Students by Department:")
print(dept_student_count)

highest_marks = c['Total_Marks'].max()
lowest_marks = c['Total_Marks'].min()
print("Highest Marks:", highest_marks)
print("Lowest Marks:", lowest_marks)

pass_percentage = (c['Result'] == 'Pass').sum() / len(c) * 100
print("Pass Percentage:", pass_percentage)

best_dept = dept_avg_marks.idxmax()
print("Department with Highest Average Marks:", best_dept)

top_three_students = c.nlargest(3, 'Total_Marks')
print("Top Three Students:")
print(top_three_students)

