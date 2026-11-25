# EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING: 
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy import stats
print("Libraries imported successfully")
data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

print("Dataset loaded successfully")
data.head()
```
<img width="972" height="232" alt="image" src="https://github.com/user-attachments/assets/33996131-6e51-435e-9dfe-7d59e83a1a54" />

```
data['fare'] = data['fare'].fillna(data['fare'].mean())
```

```
fare_data = data[data['fare'] > 0].copy()
```

```
fare_data['log_fare'] = np.log(fare_data['fare'])
```

```
fare_data['reciprocal_fare'] = 1 / fare_data['fare']
```

```
fare_data['sqrt_fare'] = np.sqrt(fare_data['fare'])
```

```
fare_data['square_fare'] = np.square(fare_data['fare'])
```

```
fare_data['boxcox_fare'], _ = stats.boxcox(fare_data['fare'])
```

```
pt = PowerTransformer(method='yeo-johnson')
fare_data['yeojohnson_fare'] = pt.fit_transform(fare_data[['fare']])
```

```
print("Feature Transformation Completed")
fare_data.head()
```
<img width="1560" height="232" alt="image" src="https://github.com/user-attachments/assets/afdf0c02-aa48-46f8-8a11-764b61554f5a" />

```
fare_data.to_csv('feature_transformed_data.csv', index=False)
print("File saved successfully!")
```

```
File saved successfully!
```

```
print("Final Transformed Dataset:")
display(fare_data.head())

print("\nDataset Shape:", fare_data.shape)
print("\nFinal Columns:")
print(fare_data.columns)
```

<img width="1537" height="382" alt="image" src="https://github.com/user-attachments/assets/03e027f6-8204-4741-bfa8-58d47c2215d3" />

# RESULT:

Thus the program to implement the linear regression using gradient descent is written and verified using 
Python programming. 

       
