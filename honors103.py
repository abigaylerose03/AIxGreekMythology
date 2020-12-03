"""
@author Abigayle Peterson
@date 11/21/20

"""

import pandas as pd
import numpy as np
import json
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import numpy as np
import pickle


df = pd.read_csv('all.csv', nrows=10)
print(df.head())

df.target_name = 'immortal'
print(df.target_name)

df.replace(to_replace='NaN',value='N/A',inplace=True)
df['relatives__father'].fillna('N/A', inplace=True)

df.replace(to_replace='NaN',value='N/A',inplace=True)
df['relatives__mother'].fillna('N/A', inplace=True)

df.replace(to_replace='NaN',value='N/A',inplace=True)
df['descritpion'].fillna('N/A', inplace=True)

df.replace(to_replace='NaN',value='N/A',inplace=True)
df['greekname'].fillna('N/A', inplace=True)

df.replace(to_replace='NaN',value=0,inplace=True)
df['gender'].fillna(0, inplace=True)


for i in df["status"]:
	if i == "immortal":
		df.replace("immortal", value=1, inplace=True)
	else:
		df.replace("mortal", value=0, inplace=True)

for i in df["category"]:
	if i == "major olympians":
		df.replace("major olympians", value=1000, inplace=True)
	elif i == 'twelve titan':
		df.replace("twelve titan", value=900, inplace=True)
	elif i == 'titan':
		df.replace("titan", value=800, inplace=True)
	elif i == 'minor figure':
		df.replace("minor figure", value=10, inplace=True)
	elif i == 'giant':
		df.replace("giant", value=200, inplace=True)
	elif i == 'sea deity':
		df.replace("sea deity", value=230, inplace=True)
	elif i == 'sky deity':
		df.replace("sky deity", value=220, inplace=True)
	elif i == 'rustic deity':
		df.replace("rustic deity", value=220, inplace=True)
	elif i == 'king':
		df.replace("king", value=100, inplace=True)
	elif i == "creature":
		df.replace("creature", value=80, inplac=True)
	else:
		print("None.")

for i in df["gender"]:
	if i == 'male':
		df.replace("male", value=1.0, inplace=True)
	elif i == 'female':
		df.replace("female", value=2.0, inplace=True)
	else:
		print("None.")

for i in df["isgod"]:
	if i == "yes":
		df.replace("yes", value=1, inplace=True)
	elif i == "no":
		df.replace("no", value=0, inplace=True)
	else:
		print("None.")


print(df["isgod"])


x = df[['category', 'status', 'gender', 'defense', 'attack']]
y = df[['isgod']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
# This is where we set our machine learning algorithm type - Decision Tree
model = DecisionTreeClassifier(criterion='gini',random_state=40) 
model.fit(x_train,y_train)


# Use the model to make predictions using our testing input data
pickle.dump(model, open('model.pkl', 'wb')) # wb means written in binary just a reminder lol
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[300, -100, 0, 1000, 100]]))



# Use the model to make predictions using our testing input data
y_pred = model.predict(x_test)
print(y_pred)
# Calculate the accuracy of the model as a percent
accuracy = accuracy_score(y_test,y_pred)*100
print("Accuracy: " + str(accuracy))


# This is where we set our machine learning algorithm type - Naive Bayes


"""
 The variable new_pokemon represents the unknown pokemon with random stats
 0 means not legendary, 1 means legendary
"""
new_test = np.array([30, -100, 0, 1000, 100]).reshape(1,-1)

if model.predict(new_test) == [1]:
  print("You would most likely be a god!")
else:
  print("You would most likely not be a god!")


# X = X.reshape(X.shape[1:])
# X = X.transpose()


# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.75, random_state=45)









# X, y = make_blobs(n_samples=1000, centers=2, n_features=6, random_state=2)

# print(X.shape, y.shape) # print state vectors 
# model = LogisticRegression(solver='lbfgs')

# model.fit(X, y)
# yhat = model.predict(X)

# acc = accuracy_score(y, yhat)
# print(acc)

# new_input = [[2.1094014124, -1.710284124, 13.512512, 1.12509501, 1.1209402, 5.32002]]
# new_output = model.predict(new_input)
# print(new_input, new_output)

# if new_output == 1:
# 	print("You are a god")
# else:
	# print("You are most likely not a god.")


# print(df.target)
# print(df.head)
# print(df['immortal'])

