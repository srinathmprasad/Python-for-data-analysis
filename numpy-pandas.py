#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:55:28 2020

@author: Srinath
"""

### NumPy

import numpy as np

# one dimensional array
np.array([1,2,3])

# two-dimensional array
np.array([[1,2,3],[4,5,6]])

np.zeros((3,4), dtype=int)

np.ones((3,4), dtype=int)

np.full((3,4), fill_value=10, dtype = float)

np.empty((3,4))

np.eye(5, dtype=int)

# arange produces an ARRAY (instead of a list)
np.arange(2,10, 0.5)

# for non-integers
np.linspace(1,10, 5)



## Arrays with random elements

# uniformly distributed between 0 and 1
np.random.random((3,4))

np.random.normal(0,1,(3,4))

np.random.randint(2,10,(3,4))

np.random.seed(123)
np.random.randint(0,5,(2,5))

np.random.random(10)


## Array type and attributes

a = np.array([[1,2,3],[3,4,5]])

a.ndim

a.shape

a.size

a.dtype

def info(name, a):
    print(f"{name} has dim {a.ndim}, shape {a.shape}, size {a.size}, and dtype{a.dtype}:")
    print(a)

b=np.array([[1,2,3], [4,5,6]])
info("b", b)

c=np.array([b, b]) # Creates a 3-dimensional array
info("c", c)

d=np.array([[1,2,3,4]]) # a row vector
info("d", d)

## Slicing

a = np.array([1,4,2,7,9,5])

a[::-1] # reverses the array

## Reshaping

a=np.arange(9)

anew = a.reshape(3,3)

info("anew", anew)

d=np.arange(4)

bc = b[:,0].reshape(2,1) # column vector
br = b[:,0].reshape(1,2) # row vector


###############
cr=np.arange(1,10).reshape(3,3)
c=np.arange(1,5).reshape(2,2)

arr = np.concatenate((c,c),axis=1)
parts = np.split(arr, 2 , axis=0)


def get_row_vectors(x):
    parts_row = np.split(x, 2, axis=0)
    return parts_row


def get_col_vectors(x):
    parts_col = np.split(x, 4, axis=1)
    return parts_col

get_row_vectors(arr)
    

get_col_vectors(arr)


np.arange(3) + np.array([4])

#############

import pandas as pd
a=pd.read_csv("https://www.cs.helsinki.fi/u/jttoivon/dap/data/fmi/kumpula-weather-2017.csv")['Air temperature (degC)'].values

a.shape

np.sum(a<0)

c = a > 0
c[:10]


b=np.arange(16).reshape(4,4)
print(b)
row=np.array([0,2])
col=np.array([1,3])
b[row, col]


## Sorting of arrays

b=np.random.randint(0,10, (4,4))

np.sort(b,axis=0)
np.sort(b,axis=1)


np.argsort(b, axis=1) # provides the indices of sorted elements


## Matrix operations

np.random.seed(12)
a = np.random.randint(1,10,(3,2))
b = np.random.randint(1,10,(2,5))
c = np.matmul(a,b)
c = a@b


### matplotlib


import matplotlib.pyplot as plt

y1=np.array([2, 5, 7, 4, 7, 0, 3, 1, 9, 2])
x1 = np.array([1,3,5,7,9,11,13,15,17,19])

y2 = np.array([4,3,2,4,5,6,8,9,10,1])
x2 = np.arange(10)
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.title("Test figure")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

fig, ax = plt.subplots(2,2)
print(ax.shape)
print(np.arange(6))
ax[0,0].plot(np.arange(6))
ax[0,1].plot(np.arange(6,0,-1))
ax[1,0].plot((-1)**np.arange(6))
ax[1,1].plot((-1)**np.arange(1,7))
plt.show()





# ex 10

a = np.array([[1,2,3,15],[3,4,7,100],[5,6,7,50],[8,4,12,25]])


def subfigures(a):
    x = a[:,0]
    y = a[:,1]
    fig, ax = plt.subplots(1,2)
    ax[0].plot(x,y)
    ax[1].scatter(x,y,c=a[:,2],s=a[:,3])
    plt.show()
    
subfigures(a)


### Pandas

import pandas as pd

wh = pd.read_csv("https://www.cs.helsinki.fi/u/jttoivon/dap/data/fmi/kumpula-weather-2017.csv")
wh.head()   # The head method prints the first 5 rows

wh.shape    # Dimensions

wh.columns

wh["Snow depth (cm)"].head()

wh["Air temperature (degC)"].mean()

wh.drop("Time zone",axis=1).head()    # Return a copy with one column removed, the original DataFrame stays intact

wh.head()

wh["Rainy"] = wh["Precipitation amount (mm)"]>5

wh.head()

# Series - one dimensional version of a dataframe

s=pd.Series([1, 4, 5, 2, 5, 2])

s.dtype

wh.dtypes

wh.name = "helsinki"

s2 = s[[0,5]]


t = s[2]

s.values # convert to numpy array

s4 = pd.Series(["Jack", "Jones", "James"], index=[1,2,3]) 

print(s4[1]) # will use refer to explicit index and results "Jack"

print(s4.loc[1]) # uses explicit index and results "Jack"

print(s4.iloc[1]) # uses implicit index and results  "Jones"

## Dataframe indexing and Slicing

a_df = pd.DataFrame(a, columns=["first","second","third", "fourth"],index=["one","two","three","four"])
# note: it is 'index' keyword and not 'rows'

a_df.index

a_df.columns

df = pd.DataFrame([[1000, "Jack", 21], [1500, "John", 29]],columns=["Wage", "Name", "Age"])

s1 = pd.Series([1,2,3])
s2 = pd.Series([4,5,6], name="b")

pd.DataFrame({"a":s1,"b":s2})


wh[["Time zone","Time"]]

wh[0:2]

wh.loc[1,"Time zone"]

df = pd.DataFrame([[1000, "Jack", 21], [1500, "John", 29]], columns=["Wage", "Name", "Age"])


df.dtypes




df.loc[1,["Wage","Name"]]

a = np.random.randint(1,100,(3,4))
data=pd.read_csv("https://www.cs.helsinki.fi/u/jttoivon/dap/data/fmi/kumpula-weather-2017.csv")


d = pd.DataFrame(a,columns=["first","second","third","fourth"],index=["one","two","three"])

# With iloc everything works like with NumPy arrays: indexing, slicing, 
# fancy indexing, masking and their combinations. With loc it is the 
# same but now the names in the explicit indices are used for specifying 
# rows and columns.

ds = d.loc[["one","two"],["first","second"]] # using loc
# this is equivalent to
d.iloc[0:2,0:2] #using iloc

t = data.loc[0:11,["Year","m","d"]] # using loc

t = data.iloc[0:11,[0,1,2]] # using iloc

# if you are just using column names, it seems you can slice df without using loc or iloc
data[["Year","m","d"]]

data2 = data.drop(["Year", "m", "d"], axis=1) 


# Sufficient stats

data2.mean()

data2["Snow depth (cm)"].describe()["max"]

# Missing values

data[["Precipitation amount (mm)","Snow depth (cm)","Air temperature (degC)"]]

data["Snow depth (cm)"].unique()
data["Snow depth (cm)"].isnull().unique()
data["Snow depth (cm)"].notnull()
data["Snow depth (cm)"].fillna("Missing").unique()

# to get the null values and corresponding rows with indices
data[data.isnull().any(axis=1)]

data[data["Air temperature (degC)"]==12.6]


# to drop the rows containing null values
data.dropna().shape

# to drop the columns containing null values
data.dropna(axis=1).shape

data.dropna

# converting colums to different types

data2=data.copy()
data2["Snow depth (cm)"]=data["Snow depth (cm)"].fillna(method='ffill') #  ffill fills the n/a values with previous record value in the column
data2 = data2.astype({"Snow depth (cm)":int,"Time zone":str})
data2["Snow depth (cm)"].unique()

# where and mask methods used like if else to replace values based on condition
# where - Replace values where the condition is False.
data2["Snow depth (cm)"] = pd.DataFrame(data2["Snow depth (cm)"].where(data2["Snow depth (cm)"]<10,20))
data2["Snow depth (cm)"].unique()

# mask - Replace values where the condition is True.
data2["Snow depth (cm)"] = pd.DataFrame(data2["Snow depth (cm)"].mask(data2["Snow depth (cm)"]>10,20))
data2["Snow depth (cm)"].unique()



# IMPORTANT
# nested loops in single line

l = []
d = {}
for c in "AB":
    for i in [2,3]:
        l.append(str(c)+str(i))
    d.update({c:l})
    l = []   
df = pd.DataFrame(d,[2,3])  
print(df)      

# alterantively

data = {c : [str(c) + str(i) for i in [2,3]] for c in "AB"}

def makedf(cols,ind):
    data = {c : [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data,ind)


a = makedf("AB",[0,1])
b = makedf("AB",[2,3])
c = makedf("CD",[0,1])
# concatenating dataframes

pd.concat([a,b])
pd.concat([a,a])
pd.concat([a,a],ignore_index=True)
pd.concat([a,a],ignore_index=True,axis=1)
pd.concat([a,c],ignore_index=False,axis=1)

# Merging dataframes

df = pd.DataFrame([[1000, "Jack", 21], [1500, "John", 29]], columns=["Wage", "Name", "Age"])
df2 = pd.DataFrame({"Name" : ["John", "Jack"], "Occupation": ["Plumber", "Carpenter"]})

pd.merge(df,df2)

df3 = pd.concat([df2,pd.DataFrame({"Name":["James"], "Occupation":["Painter"]})],ignore_index=True)

pd.merge(df,df3) # inner join by default

pd.merge(df,df3,how="outer") # outer join 

books = pd.DataFrame({"Title" : ["War and Peace", "Good Omens", "Good Omens"] ,
                      "Author" : ["Tolstoi", "Terry Pratchett", "Neil Gaiman"]})


collections = pd.DataFrame([["Oodi", "War and Peace"],
                           ["Oodi", "Good Omens"],
                           ["Pasila", "Good Omens"],
                           ["Kallio", "War and Peace"]], columns=["Library", "Title"])


libraries_with_books_by = pd.merge(books, collections)


# rename column names

wh3 = wh.rename(columns={"m": "Month", "d": "Day", "Precipitation amount (mm)" : "Precipitation",
                         "Snow depth (cm)" : "Snow", "Air temperature (degC)" : "Temperature"})
wh3.head()

groups = wh3.groupby("Month")

for key, group in groups:
    print(key, len(group))
    

groups.get_group(3)

groups[["Temperature","Precipitation"]].mean()
    

wh3["Snow"].unique()

# replace a value in a colum with new value
wh3.loc[wh3.Snow==13,"Snow"]= 130
# dataframe.loc[<condition>,<column>] = new value

# Filtering groups 

def myfilter(df):      # The filter function must return a boolean value
    return df["Precipitation"].sum() >= 150

wh4.groupby("Month").filter(myfilter)    # Filter out months with total precipitation less that 150 mm


# Filter out months with mean temperature less than 0
def filter2(df):
    return df["Temperature"].mean()<=0
    
wh3.groupby("Month").filter(filter2)
    
wh3["Date"]=pd.DataFrame(wh3["Year"])
wh3["Date"]=pd.to_datetime(wh3[["Year","Month","Day"]])    


# Machine Learning

import sklearn   # This imports the scikit-learn library


np.random.seed(0)
n=20   # Number of data points
x=np.linspace(0, 10, n)
y=x*2 + 1 + 1*np.random.randn(n) # Standard deviation 1
print(x)
print(y)


from sklearn.linear_model import LinearRegression


model=LinearRegression(fit_intercept=True)
model.fit(x[:,np.newaxis], y)
xfit=np.linspace(0,10,100)
yfit=model.predict(xfit[:, np.newaxis])
plt.plot(xfit,yfit, color="black")
plt.plot(x,y, 'o')


import seaborn as sns


from sklearn.datasets import make_blobs
X,y = make_blobs(100, 2, centers=3, random_state=2, cluster_std=1.5)
colors=np.array(["red", "blue", "green"])
plt.scatter(X[:, 0], X[:, 1], c=colors[y], s=50, label=colors)
plt.legend();

for color in colors:
    plt.scatter(X[:, 0], X[:, 1], c=colors[y], s=50, label=colors)
plt.legend();
plt.show()


from sklearn.datasets import make_blobs
X,y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
colors=np.array(["red", "blue"])
plt.scatter(X[:, 0], X[:, 1], c=colors[y], s=50)
for i, color in enumerate(colors):
    plt.scatter([],[], c=color, s=50, label=i)
plt.legend();
plt.show()




df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])

df.apply(lambda x: [1, 2], axis=0)



dict_a = [{'name': 'python', 'points': 10}, {'name': 'java', 'points': 8}]


# =============================================================================
# lambda arguments: expression

# can have any number of arguments and ONLY one expression
#

# map(function_object, iterable)


# filter(function_object, iterable) 

# the function_object here returns a boolean value 
# and is called for each element of the iterable
# and returns elements only for which the function returns True

# =============================================================================


list(map(lambda x: x+2, df[:1]))

df[['A','B']].apply(lambda x: x+2)
df.apply(lambda x: x+2)
df.applymap(lambda x: x+2)
df['A'].map(lambda x: x+2)


df2 = pd.DataFrame(np.random.randint(1,9, (3,4)))


df2["new3"] = df2.apply(lambda x: x[0]+x[1]-x[3], axis=1)
df2["new"] = df2[0]+df2[1]-df2[3]

import pandas as pd
import matplotlib.pyplot as plt



