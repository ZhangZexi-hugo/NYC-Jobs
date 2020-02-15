import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nose import selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, f_regression, chi2, SelectKBest, SelectPercentile, SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import preprocessing
import pandas as pd
from sodapy import Socrata
from sklearn import tree


def datafrom():
    client = Socrata("data.cityofnewyork.us", None)
    results = client.get("kpav-sd4t", limit=3000)
    results_df = pd.DataFrame.from_records(results)
    print(results_df.info())
    return results_df


def clean_data(data):
    print(data.describe())
    print(data.info())
    print(data.isnull().sum())
    data [['salary_range_to','salary_range_from','number_of_positions']]= data[['salary_range_to','salary_range_from','number_of_positions']].astype(float)
    # jOb Description, Minimum Qual Requirements, and preferred skills are not able to be processed in this case
    data.drop(['job_description','minimum_qual_requirements','to_apply','preferred_skills','process_date','hours_shift','work_location','work_location_1','post_until','additional_information'], axis = 1, inplace = True)
    data.dropna(subset=['job_category','career_level'], inplace=True)
    data['full_time_part_time_indicator'].fillna(data['full_time_part_time_indicator'].mode()[0], inplace =True)
    data['average_salary'] = (data['salary_range_to'] + data['salary_range_from']) / 2
    data.drop(['salary_range_to','salary_range_from', 'job_id'], axis =1,inplace =True)
    print(data.info())
    for i in range(len(data)):
        if data['salary_frequency'].iloc[i] =='Hourly':
            data['average_salary'].iloc[i] = (data['average_salary'].iloc[i])*1950
        elif data['salary_frequency'].iloc[i] =='Daily':
            data['average_salary'].iloc[i] = (data['average_salary'].iloc[i])*260

    data.loc[(data['salary_frequency']=='Hourly'),'salary_frequency'] ='Annual'
    data.loc[(data['salary_frequency'] == 'Darily'), 'salary_frequency'] = 'Annual'
    return data
def level_dist(data):
    level_d = pd.concat([data['career_level'],data['number_of_positions']],axis = 1)
    level_d = level_d.groupby(level_d['career_level']).sum()
    level_d.plot(kind='bar',stacked=True, colormap = 'Paired')
    plt.title('distrbution of career levels')
    plt.xlabel('Career Type')
    plt.ylabel('Counts')
    plt.show()

def level_s(data):
    levels = data[['number_of_positions','agency']][data['career_level']=='Student']
    levels = levels.groupby(levels['agency']).sum().sort_values(['number_of_positions'], ascending=False)
    levels['agency'] = levels.index
    sns.barplot(x=levels.iloc[:5, 0], y=levels.iloc[:5, 1])
    plt.title('Top Five Agency posts of Student')
    plt.xlabel('Number of position for student based on agency')
    plt.ylabel('Counts')
    plt.show()

def level_experienced(data):
    exp = data[['number_of_positions', 'agency']][data['career_level'] == 'Experienced (non-manager)']
    exp = exp.groupby(exp['agency']).sum().sort_values(['number_of_positions'], ascending=False)
    exp['agency'] = exp.index
    print(exp.describe())
    sns.barplot(x=exp.iloc[:5,0],y=exp.iloc[:5,1])
    plt.title('Top Five Agency posts of Experienced (non-manager)')
    plt.xlabel('Number of position for Experienced (non-manager) based on agency')
    plt.ylabel('Counts')
    plt.show()

def level_entry(data):
    entry=data[['number_of_positions', 'agency']][data['career_level'] == 'Entry-Level']
    entry = entry.groupby(entry['agency']).sum().sort_values(['number_of_positions'], ascending=False)
    entry['agency'] = entry.index
    print(entry.describe())
    sns.barplot(x=entry.iloc[:5,0],y=entry.iloc[:5,1])
    plt.title('Top Five Agency posts of Entry-Level')
    plt.xlabel('Number of position for Entry-Level based on agency')
    plt.ylabel('Counts')
    plt.show()

def level_manager(data):
    manager=data[['number_of_positions', 'agency']][data['career_level'] == 'Manager']
    manager = manager.groupby(manager['agency']).sum().sort_values(['number_of_positions'], ascending=False)
    manager['agency'] = manager.index
    print(manager.describe())
    sns.barplot(x=manager.iloc[:5,0],y=manager.iloc[:5,1])
    plt.title('Top Five Agency posts of manager')
    plt.xlabel('Number of position for Manager based on agency')
    plt.ylabel('Counts')
    plt.show()

def level_exec(data):
    executive=data[['number_of_positions', 'agency']][data['career_level'] == 'Executive']
    executive = executive.groupby(executive['agency']).sum().sort_values(['number_of_positions'], ascending=False)
    executive['agency'] = executive.index
    print(executive.describe())
    sns.barplot(x=executive.iloc[:5,0],y=executive.iloc[:5,1])
    plt.title('Top Five Agency number of positions of Executive')
    plt.xlabel('Number of position for Executive based on agency')
    plt.ylabel('Counts')
    plt.show()

def s_ent(data):
    ent=data[['average_salary', 'agency']][data['career_level'] == 'Entry-Level']
    print(ent.describe())
    ent = ent.groupby(ent['agency']).mean().sort_values(['average_salary'], ascending=False)
    ent['agency'] = ent.index
    sns.barplot(x=ent.iloc[:5,0],y=ent.iloc[:5,1])
    plt.vlines(53556, -0.5, 5,colors="r", linestyles="dashed")
    plt.title('Top Five Agency salary of Entry-Level')
    plt.xlabel('average salary for Entry-Level based on agency')
    plt.ylabel('average_salary')
    plt.show()

def s_student(data):
    student1=data[['average_salary', 'agency']][data['career_level'] == 'Student']

    student1 = student1.groupby(student1['agency']).mean().sort_values(['average_salary'], ascending=False)
    print(student1)
    student1['agency'] = student1.index
    print(student1.describe())
    sns.barplot(x=student1.iloc[:6,0],y=student1.iloc[:6,1])
    plt.vlines(35577, -0.5, 5,colors="r", linestyles="dashed")
    plt.title('Top Five Agency salary of Student')
    plt.xlabel('average Student salary for Student based on agency')
    plt.ylabel('average_salary')
    plt.show()

def s_manager(data):
    manager1=data[['average_salary', 'agency']][data['career_level'] == 'Manager']
    #print(manager1.describe())
    manager1 = manager1.groupby(manager1['agency']).mean().sort_values(['average_salary'], ascending=False)
    manager1['agency'] = manager1.index
    print(manager1.describe())
    sns.barplot(x=manager1.iloc[:5,0],y=manager1.iloc[:5,1])
    plt.vlines(95511, -0.5, 5,colors="r", linestyles="dashed")
    plt.title('Top Five Agency salary of manager')
    plt.xlabel('average_salary for manager based on agency')
    plt.ylabel('average_salary')
    plt.show()

def s_exp(data):
    exp1=data[['average_salary', 'agency']][data['career_level'] == 'Experienced (non-manager)']
    #print(exp1.describe())
    exp1 = exp1.groupby(exp1['agency']).mean().sort_values(['average_salary'], ascending=False)
    exp1['agency'] = exp1.index
    print(exp1.describe())
    sns.barplot(x=exp1.iloc[:5,0],y=exp1.iloc[:5,1])
    plt.vlines(73460, -0.5, 5,colors="r", linestyles="dashed")
    plt.title('Top Five Agency salary of Experienced')
    plt.xlabel('average_salary for Experienced based on agency')
    plt.ylabel('average_salary')
    plt.show()

def s_exe(data):
    exe1=data[['average_salary', 'agency']][data['career_level'] == 'Executive']
    #print(exe1.describe())
    exe1 = exe1.groupby(exe1['agency']).mean().sort_values(['average_salary'], ascending=False)
    exe1['agency'] = exe1.index
    print(exe1.describe())
    sns.barplot(x=exe1.iloc[:5,0],y=exe1.iloc[:5,1])
    plt.vlines(139556, -0.5, 5,colors="r", linestyles="dashed")
    plt.title('Top Five Agency salary of Executive')
    plt.xlabel('average_salary for Executive based on agency')
    plt.ylabel('average_salary')
    plt.show()

def timeserise(data):
    data['posting_date']=pd.to_datetime(data['posting_date'])
    td = data[(data['posting_date']< '1/1/2020') | (data['posting_date']>= '2/12/2015')]
    td['Month'] = td['posting_date']
    postsbymonth = pd.concat([td['Month'],td['number_of_positions']],axis = 1)
    postsbymonth['Month']=td['Month'].dt.strftime('%Y')
    postsbymonth=postsbymonth.groupby('Month').sum()
    plt.title('The Average Monthly Posts From 2015 to 2019')
    plt.xlabel('Years')
    plt.ylabel('Posts')
    plt.plot(postsbymonth)
    plt.show()

def model_feature(data):
    X = data[['agency','posting_type','number_of_positions','title_classification','full_time_part_time_indicator',
             'career_level','level','average_salary']]
    print(data.average_salary.describe())
    salary_level =[]
    for i in X['average_salary']:
        if i <=60000:
            salary_level.append('1low_salary')
        elif i >60000 and i <= 90000:
            salary_level.append('2Mid_salary')
        elif i >=90000:
            salary_level.append('3high_salary')
    data1 = pd.concat([X.reset_index(drop=True), pd.DataFrame(salary_level, columns=['salary_level'])], axis=1)
    data2 =data1
    le = preprocessing.LabelEncoder()
    le.fit(['1low_salary','2Mid_salary','3high_salary'])
    labe= le.transform(data1['salary_level'])
    labe=pd.DataFrame(labe,columns=['labled_salary'])
    number = data1['number_of_positions']
    data2.drop(['average_salary','number_of_positions','salary_level'],axis = 1, inplace = True)
    data2_dummy= pd.get_dummies(data2)
    data_final = pd.concat([data2_dummy,number,labe], axis = 1)
    print(data_final.info())
    return data_final


def x_y (data):
    Y=data['labled_salary']
    X=data.iloc[:,0:81]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
    return x_train,x_test,y_train,y_test

def train_test(x_train,x_test,y_train,y_test):
    select = SelectPercentile(percentile=75)
    select.fit(x_train,y_train)
    x_train_selected = select.transform(x_train)
    f,p = f_classif(x_train,y_train)
    plt.figure()
    plt.plot(p, 'o')
    plt.xlabel('Counts of Features')
    plt.ylabel('F-score')
    plt.title('The sample distribution of F-Test')
    plt.show()
    mask = select.get_support()
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.title('Feature selection percentage')
    plt.show()
    x_test_selected = select.transform(x_test)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train_selected, y_train)
    tree_pre=clf.predict(x_test_selected)
    tree.plot_tree(clf)
    plt.title('Tree model plot')
    plt.show()
    print("Decision Tree Classifier Accuracy - " + str(100 * accuracy_score(tree_pre, y_test)) + "%")

    ra = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=50,random_state=0)
    ra.fit(x_train_selected, y_train)
    ra_pre = ra.predict(x_test_selected)
    print("Random Forest Accuracy - " + str(100 * accuracy_score(ra_pre, y_test)) + "%")




def model_select (x_train,x_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
    select.fit(x_train, y_train)
    X_train_l1 = select.transform(x_train)
    print(x_train.shape)
    print(X_train_l1.shape)



