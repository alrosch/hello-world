## CREATED WITH JUPYTER NOTEBOOK ##

# coding: utf-8

# In[1]:


# import libraries needed
import pandas as pd
import numpy as np


# In[2]:


# read csv files to be able to get hands on the data
train_ds = pd.read_csv(".../HP_train_dataset.csv")
test_ds = pd.read_csv(".../HP_test_dataset.csv")


# In[3]:


# Make a copy of both datasets, so the original datasets don't get lost
train = train_ds.copy()
test = test_ds.copy()

#Print the first rows of the training dataset
train.head()


# In[4]:


# Track which columns have "NaN" or 0 values
train.info()


# In[5]:


# Set the target data
y = train["SalePrice"]


# In[6]:


# Remove unnecessary data
train = train.drop(["SalePrice", "Id"], axis = 1);


# In[7]:


test = test.drop(["Id"], axis = 1)


# In[8]:


#concatenate both datasets to speedup cleaning/filling data
data = [train, test]
data = pd.concat(data)


# In[9]:


data.info()


# In[10]:


# Erase data which has around 50% missing values
data = data.drop(["MiscFeature", "PoolQC", "Fence", "Alley", "FireplaceQu"], axis = 1)


# In[11]:


data.info()


# In[12]:


# Fill missing numerical data with the median
missing_num_data = ["LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", 
                    "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", "GarageCars", "GarageArea"]
for column in missing_num_data:
    data[column] = data[column].fillna(data[column].median())


# In[13]:


data.info()


# In[14]:


#Drop Categorical Data ((Didn't find out how to transform caregorical data into integers!))
missing_cat_data = ["GarageType", "GarageFinish", "GarageQual", "GarageCond", "SaleType", "Functional", "Utilities",
              "Electrical", "Functional", "KitchenQual", "Exterior1st", "Exterior2nd", "MasVnrType", "BsmtQual",
              "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MSZoning"]
data = data.drop(missing_cat_data, axis=1)


# In[15]:


# Transform categorical data with missing data into integers ((DIDN'T WORK!))
# Fill missing categorical data with the most repeated value/column (mode)
#for column in missing_cat_data:
    #_, data[column] = np.unique(data[column], return_inverse=True) 
    #data[column] = data[column].fillna(data[column].mode())

# Normalize numerical data
for column in list(data.select_dtypes(exclude=[object])):
    data[column] = (data[column] - data[column].mean()) / data[column].std()


# In[16]:


# Transform categorical data into integers
for column in list(data.select_dtypes(include=[object])):
    _, data[column] = np.unique(data[column], return_inverse=True)


# In[17]:


data.info()


# In[18]:


# Normalize 
y = (y - y.mean()) / y.std()


# In[19]:


#Split the data
X_train = data[:1460]
X_test = data[1460:]
Y = y


# In[22]:


#Create the Neural Network model with Keras

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(112, activation="relu", input_shape=(56,)))
model.add(Dense(56, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1,))
          
model.compile(loss='mean_squared_error', optimizer='adam')


# In[23]:


model.fit(X_train.values, Y.values, epochs=50, batch_size=1, verbose=1)


# In[25]:


#Predict Prices for the test set
predictions = model.predict(X_test.values)


# In[28]:


# "Unnormalize" target data
predictions = predictions * train_ds["SalePrice"].std() + train_ds["SalePrice"].mean()


# In[34]:


#Create the requiered .csv file format
solution = pd.DataFrame({"Id": test_ds["Id"], "SalePrice": predictions.flatten()})
solution.to_csv("Solution.csv",index = False)
## Kaggle score = 0.13363 (1708/4551) Top 37.5%

