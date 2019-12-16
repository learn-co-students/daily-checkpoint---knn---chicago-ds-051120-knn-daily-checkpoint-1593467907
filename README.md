```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
```

### 1. What does the '$k$' represent in "$k$-Nearest Neighbors"?


```python
# Your answer here


```

### 2. How do the variance and bias of my model change as I adjust $k$? What would happen if I set $k$ to $n$, the size of my dataset?


```python
# Your answer here


```

### $k$-Nearest Neighbors in Scikit-Learn


```python
wine = load_wine()
```


```python
print(wine.DESCR)
```


```python
wine.feature_names
```


```python
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df.head()
```


```python
wine.target[:5]
```

### 3. Perform a train-test split on the data (including the target), and
### fit the $k$-Nearest Neighbors Classifier to the training data with $k$ = 7.


```python
# Your code here


```

### Confusion Matrix


```python
confusion_matrix(y_test, knn.predict(X_test))
```

### 4. How accurate is the model?  What is the precision of the model in classifying wines from *Class 1*?


```python
# Your answer here


```

### Now try a model with $k$ = 5 and a Manhattan distance metric. (You can use the same train-test split.)

### 5. How accurate is the new model? What is the recall of the model in classifying wines from *Class 0*?


```python
# Your answer here


```


```python

```
