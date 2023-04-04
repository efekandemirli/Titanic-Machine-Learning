# Titanic Machine Learning
 Titanic Machine Learning Challenge Homework





![read1](https://user-images.githubusercontent.com/84057714/229853528-b0982407-443d-47d3-88bf-568f0486647e.jpg)
![read2](https://user-images.githubusercontent.com/84057714/229853554-e9aa6b43-e352-42b5-b5f5-803f2d0da3e7.jpg)




<class 'pandas.core.frame.DataFrame'>
Int64Index: 712 entries, 58 to 354
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  712 non-null    int64  
 1   Survived     712 non-null    int64  
 2   Pclass       712 non-null    int64  
 3   Name         712 non-null    object 
 4   Sex          712 non-null    object 
 5   Age          563 non-null    float64
 6   SibSp        712 non-null    int64  
 7   Parch        712 non-null    int64  
 8   Ticket       712 non-null    object 
 9   Fare         712 non-null    float64
 10  Cabin        161 non-null    object 
 11  Embarked     710 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 72.3+ KB
None







    PassengerId  Survived  Pclass       Age  ...    S    Q  Female  Male
58            59         1       2   5.00000  ...  0.0  1.0     1.0   0.0
498          499         0       1  25.00000  ...  0.0  1.0     1.0   0.0
48            49         0       3  30.12286  ...  0.0  0.0     0.0   1.0
732          733         0       2  30.12286  ...  0.0  1.0     0.0   1.0
240          241         0       3  30.12286  ...  0.0  0.0     1.0   0.0
..           ...       ...     ...       ...  ...  ...  ...     ...   ...
270          271         0       1  30.12286  ...  0.0  1.0     0.0   1.0
455          456         1       3  29.00000  ...  0.0  0.0     0.0   1.0
798          799         0       3  30.00000  ...  0.0  0.0     0.0   1.0
654          655         0       3  18.00000  ...  1.0  0.0     1.0   0.0
354          355         0       3  30.12286  ...  0.0  0.0     0.0   1.0

[712 rows x 12 columns]




<class 'pandas.core.frame.DataFrame'>
Int64Index: 712 entries, 58 to 354
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  712 non-null    int64  
 1   Survived     712 non-null    int64  
 2   Pclass       712 non-null    int64  
 3   Age          712 non-null    float64
 4   SibSp        712 non-null    int64  
 5   Parch        712 non-null    int64  
 6   Fare         712 non-null    float64
 7   C            712 non-null    float64
 8   S            712 non-null    float64
 9   Q            712 non-null    float64
 10  Female       712 non-null    float64
 11  Male         712 non-null    float64
dtypes: float64(7), int64(5)
memory usage: 72.3 KB
None




GridSearchCV(cv=3, estimator=RandomForestClassifier(),
             param_grid=[{'max_depth': [None, 5, 10],
                          'min_samples_split': [2, 3, 4],
                          'n_estimators': [10, 100, 200, 500]}],
             return_train_score=True, scoring='accuracy')
RandomForestClassifier(max_depth=10)
0.7988826815642458





 PassengerId  Survived  Pclass        Age  ...    S    Q  Female  Male
0              1         0       3  22.000000  ...  0.0  1.0     0.0   1.0
1              2         1       1  38.000000  ...  0.0  0.0     1.0   0.0
2              3         1       3  26.000000  ...  0.0  1.0     1.0   0.0
3              4         1       1  35.000000  ...  0.0  1.0     1.0   0.0
4              5         0       3  35.000000  ...  0.0  1.0     0.0   1.0
..           ...       ...     ...        ...  ...  ...  ...     ...   ...
886          887         0       2  27.000000  ...  0.0  1.0     0.0   1.0
887          888         1       1  19.000000  ...  0.0  1.0     1.0   0.0
888          889         0       3  29.699118  ...  0.0  1.0     1.0   0.0
889          890         1       1  26.000000  ...  0.0  0.0     0.0   1.0
890          891         0       3  32.000000  ...  1.0  0.0     0.0   1.0

[891 rows x 12 columns]




GridSearchCV(cv=3, estimator=RandomForestClassifier(),
             param_grid=[{'max_depth': [None, 5, 10],
                          'min_samples_split': [2, 3, 4],
                          'n_estimators': [10, 100, 200, 500]}],
             return_train_score=True, scoring='accuracy')
     PassengerId  Pclass       Age  SibSp  Parch  ...    C    S    Q  Female  Male
0            892       3  34.50000      0      0  ...  0.0  1.0  0.0     0.0   1.0
1            893       3  47.00000      1      0  ...  0.0  0.0  1.0     1.0   0.0
2            894       2  62.00000      0      0  ...  0.0  1.0  0.0     0.0   1.0
3            895       3  27.00000      0      0  ...  0.0  0.0  1.0     0.0   1.0
4            896       3  22.00000      1      1  ...  0.0  0.0  1.0     1.0   0.0
..           ...     ...       ...    ...    ...  ...  ...  ...  ...     ...   ...
413         1305       3  30.27259      0      0  ...  0.0  0.0  1.0     0.0   1.0
414         1306       1  39.00000      0      0  ...  1.0  0.0  0.0     1.0   0.0
415         1307       3  38.50000      0      0  ...  0.0  0.0  1.0     0.0   1.0
416         1308       3  30.27259      0      0  ...  0.0  0.0  1.0     0.0   1.0
417         1309       3  30.27259      1      1  ...  1.0  0.0  0.0     0.0   1.0

[418 rows x 11 columns]





[418 rows x 11 columns]
     PassengerId  Survived
0            892         0
1            893         0
2            894         0
3            895         0
4            896         1
..           ...       ...
413         1305         0
414         1306         1
415         1307         0
416         1308         0
417         1309         0

[418 rows x 2 columns]




