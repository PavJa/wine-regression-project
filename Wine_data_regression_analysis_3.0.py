import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # creating the heatmap
import matplotlib.pyplot as plt # data visualization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

# Step 1: Import libraries (as seen above), load the data, sort it, and check the shape and type

data = 'winequality-white.csv'

df = pd.read_csv(data,sep = ';') # data are mashed in the csv file, so I have to separate them by using ";"
#print(df.head())  # I am printing different infomration about my data to see its basic properties
#col_names = df.columns (I commented out the code, but it works)
#print(col_names) # gest names of our columns
print(df.dtypes)
print(df.shape)




# Step 2: Check for missing values

print(df.isnull()) # method n.1 gives True to any missing or null value and False otehrwise
print(df.isnull().sum()) # sums the Boolean values

print(df.isnull().values.any()) # method n.2 checks whether there is a missing value wit ha simple command




# Step 3: Define categorical variable and explore it

categorical = [var for var in df.columns if df[var].dtype=='O'] # just confirming we have no categoricl
                                                                    # variable in the original data
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n', categorical)

df['worth buying'] = np.where(df['quality'] > 6, "Yes", "No") # I create a Boolean value given the quality score.
                                                               # I assume that 7 is around 88 point on 100 scale, so yes
                                                               # it is reasonable to buy. Anything below 7 is "No",
                                                                # not worth the money.

#print(df.head()) # just checking it was added to the data frame

categorical = [var for var in df.columns if df[var].dtype=='O'] # define categorical

#print('The categorical variables is :', categorical)

print(df[categorical].isnull().sum()) # check for missing values

for var in categorical: # frequency of categorical variables
    print(df[var].value_counts())

for var in categorical: #frequency rates
    print(df[var].value_counts() / float(len(df)))







# Step 4: Explore numerical variables

numerical = [var for var in df.columns if df[var].dtype=='float64'] # define numerical as float
# (that is, all except quality which is up to 10 max )

pd.set_option('display.max_columns', None) # I want to see all columns
print(round(df[numerical].describe())) # gives me the basic statistical overview for ``numerical''

# I now check histograms for the skewness of distributions for the suspicious variables

df.columns = df.columns.str.replace(' ', '_') #replace spaces in the names of columns, so it is easier to write commands

#print(df.columns)

subfig = plt.figure(figsize=(15,10))
subfig.suptitle('Skewness of distributions of suspicious variables', fontsize=16)

plt.subplot(2, 2, 1)
fig = df.residual_sugar.hist(bins=10)
fig.set_xlabel('residual_sugar')
fig.set_ylabel('quality')

plt.subplot(2, 2, 2)
fig = df.free_sulfur_dioxide.hist(bins=10)
fig.set_xlabel('free_sulfur_dioxide')
fig.set_ylabel('quality')


plt.subplot(2, 2, 3)
fig = df.total_sulfur_dioxide.hist(bins=10)
fig.set_xlabel('total_sulfur_dioxide')
fig.set_ylabel('quality')

# finding outliers for residual sugar, free and total sulfur dioxide variables using IQR

IQR = df.residual_sugar.quantile(0.75) - df.residual_sugar.quantile(0.25)
Lower_bound = df.residual_sugar.quantile(0.25) - (IQR * 3)
Upper_bound = df.residual_sugar.quantile(0.75) + (IQR * 3)
print('Residual sugar outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_bound,
                                                                                   upperboundary=Upper_bound))

IQR = df.free_sulfur_dioxide.quantile(0.75) - df.free_sulfur_dioxide.quantile(0.25)
Lower_bound = df.free_sulfur_dioxide.quantile(0.25) - (IQR * 3)
Upper_bound = df.free_sulfur_dioxide.quantile(0.75) + (IQR * 3)
print('Free sulfur dioxide outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_bound,
                                                                                   upperboundary=Upper_bound))

IQR = df.total_sulfur_dioxide.quantile(0.75) - df.total_sulfur_dioxide.quantile(0.25)
Lower_bound = df.total_sulfur_dioxide.quantile(0.25) - (IQR * 3)
Upper_bound = df.total_sulfur_dioxide.quantile(0.75) + (IQR * 3)
print('Total sulfur dioxide outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_bound,
                                                                                   upperboundary=Upper_bound))



# Step 5: Split data for training and testing, get rid of outliers, and train the model

# First, declare feature vector X (multi-dimensional numerical values) and target variable y

X = df.drop(['worth_buying','quality'], axis=1) # I am dropping columns 'worth_buying' and 'quality' since
                                                # the first fully depends on the second, which corrupted my data
                                                # previously

y = df['worth_buying']

# Split data into separate training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape, X_test.shape)


def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable]) # limit outliers in our data using the upper bounds
                                                        # found with IQR

for df3 in [X_train, X_test]:
    df3['residual_sugar'] = max_value(df3, 'residual_sugar', 34.5)
    df3['free_sulfur_dioxide'] = max_value(df3, 'free_sulfur_dioxide', 115.0)
    df3['total_sulfur_dioxide'] = max_value(df3, 'total_sulfur_dioxide', 344.0)

print(X_train.describe()) # checking that the maximal values are now set by the thresholds



# I can now use the cleaned datat to train the model

logreg = LogisticRegression(solver='liblinear', random_state=0)

logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test) # We can predict results using our model

#print(y_pred_test) # we can print the predictions of our model (I commented out the code, but it works)










# EVALUATION of the model starts here



# Step 1: Checking accuracy score, overfitting, and underfitting

print('Model accuracy score (Test set accuracy score): {0:0.5f}'.format(accuracy_score(y_test, y_pred_test)))

y_pred_train = logreg.predict(X_train) # Compare the train-set and test-set accuracy to Check for overfitting
                                            # and underfitting
print('Training-set accuracy score: {0:0.5f}'. format(accuracy_score(y_train, y_pred_train)))

# Compare with null accuracy
count_test = y_test.value_counts() # find the most frequent class
print(count_test)

null_accuracy = (764/(764+216)) # calculate the null accuracy

print('Null accuracy score: {0:0.5f}'.format(null_accuracy))




# Step : Confusion Matrix and Heatmap


# Confusion matrix and true positives, etc. calculations
cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])


plt.figure()
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')







# Step 3: Overall indicators and classification report

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy)) # classification accuracy


classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error)) #classification error

precision = TP / float(TP + FP) # precision


print('Precision : {0:0.4f}'.format(precision))


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall)) #recall

true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))

print(classification_report(y_test, y_pred_test)) # Classification report






# Step 4: Predicted probabilities and ROC curves

# predicted probabilities

y_pred_prob = logreg.predict_proba(X_test)
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - Buy (1)', 'Prob of - Do Not Buy (0)'])
#print(y_pred_prob_df)


y_pred_buy = logreg.predict_proba(X_test)[:, 0] #probabilities for buying
#print(y_pred_buy)

y_pred_not = logreg.predict_proba(X_test)[:, 1] #probabilities for not buying
#print(y_pred_not)


# ROC curve for buying

fpry, tpry, thresholds = roc_curve(y_test, y_pred_buy, pos_label = 'Yes')

plt.figure()

plt.plot(fpry, tpry, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Buying classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')


#ROC for not buying

fprn, tprn, thresholds = roc_curve(y_test, y_pred_not, pos_label = 'Yes') #ROC for not buying

plt.figure()

plt.plot(fprn, tprn, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Not Buying classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')


# I can also plot predicted probabilities for buying to see its distribution (I commented out the code, but it works)

#plt.figure()
#plt.rcParams['font.size'] = 12

#plt.hist(y_pred_buy, bins = 10) # plot histogram with 10 bins

#plt.title('Histogram of predicted probabilities of buying') # set the title of predicted probabilities

#plt.xlim(0,1)  # set the x-axis limit

#plt.xlabel('Predicted probabilities of buying') # set the title
#plt.ylabel('Frequency')

# I can plot predicted probabilities for not buying to see its distribution (I commented out the code, but it works)
#plt.figure()
#plt.rcParams['font.size'] = 12

#plt.hist(y_pred_not, bins = 10) # plot histogram with 10 bins

#plt.title('Histogram of predicted probabilities of not buying') # set the title of predicted probabilities

#plt.xlim(0,1)  # set the x-axis limit

#plt.xlabel('Predicted probabilities of not buying') # set the title
#plt.ylabel('Frequency')






# Step 5: Correlation

X_corr = df.drop(['worth_buying'], axis=1) # drop the binary variable before doing any correlation

# this shows whol correlation heat map, but I think it is just present correlation with respect to quality,
# which I will do next (so, I commented out the code, but it works)
#plt.figure()
#plt.figure(figsize=(16, 6))
#corheat= sns.heatmap(X_corr.corr(),vmin=-1, vmax=1, annot=True)
#corheat.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);





# correlation with respect to quality

X_corr.corr()[['quality']].sort_values(by='quality', ascending=False)


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(X_corr.corr()[['quality']].sort_values(by='quality', ascending=False), vmin=-1, vmax=1,
                      annot=True, cmap='BrBG')
heatmap.set_title('Features correlating with quality', fontdict={'fontsize':18}, pad=16)

plt.show()