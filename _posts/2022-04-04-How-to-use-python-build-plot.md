---
layout: post
title: How to Make Histogram with Plotly
---
Let's build a histogram!


# Import Data


```python
import pandas as pd
from plotly import express as px

url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```

## The Basic Information of Data

We should always check waht are the variables in our data, and the total number of observations


```python
penguins.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>studyName</th>
      <th>Sample Number</th>
      <th>Species</th>
      <th>Region</th>
      <th>Island</th>
      <th>Stage</th>
      <th>Individual ID</th>
      <th>Clutch Completion</th>
      <th>Date Egg</th>
      <th>Culmen Length (mm)</th>
      <th>Culmen Depth (mm)</th>
      <th>Flipper Length (mm)</th>
      <th>Body Mass (g)</th>
      <th>Sex</th>
      <th>Delta 15 N (o/oo)</th>
      <th>Delta 13 C (o/oo)</th>
      <th>Comments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PAL0708</td>
      <td>1</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N1A1</td>
      <td>Yes</td>
      <td>11/11/07</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>MALE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Not enough blood for isotopes.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PAL0708</td>
      <td>2</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N1A2</td>
      <td>Yes</td>
      <td>11/11/07</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>FEMALE</td>
      <td>8.94956</td>
      <td>-24.69454</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PAL0708</td>
      <td>3</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N2A1</td>
      <td>Yes</td>
      <td>11/16/07</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>FEMALE</td>
      <td>8.36821</td>
      <td>-25.33302</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PAL0708</td>
      <td>4</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N2A2</td>
      <td>Yes</td>
      <td>11/16/07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adult not sampled.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAL0708</td>
      <td>5</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N3A1</td>
      <td>Yes</td>
      <td>11/16/07</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>FEMALE</td>
      <td>8.76651</td>
      <td>-25.32426</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
penguins.shape
```




    (344, 17)



## Data cleanning


```python
penguins = pd.read_csv(url)
penguins = penguins.dropna(subset = ["Body Mass (g)", "Sex"])
penguins["Species"] = penguins["Species"].str.split().str.get(0)
penguins = penguins[penguins["Sex"] != "."]

cols = ["Species", "Island", "Sex", "Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]
penguins = penguins[cols]
penguins.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Species</th>
      <th>Island</th>
      <th>Sex</th>
      <th>Culmen Length (mm)</th>
      <th>Culmen Depth (mm)</th>
      <th>Flipper Length (mm)</th>
      <th>Body Mass (g)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>MALE</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>FEMALE</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>FEMALE</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>FEMALE</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>MALE</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>190.0</td>
      <td>3650.0</td>
    </tr>
  </tbody>
</table>
</div>



We have total 344 obseravtions, and 17 variables in this data set.

## More About the Data

For example, I am interested in how many observations of different species of penguins, we could use groupby ()  to achieve it.


```python
penguins.groupby("Species").count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Island</th>
      <th>Sex</th>
      <th>Culmen Length (mm)</th>
      <th>Culmen Depth (mm)</th>
      <th>Flipper Length (mm)</th>
      <th>Body Mass (g)</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adelie</th>
      <td>146</td>
      <td>146</td>
      <td>146</td>
      <td>146</td>
      <td>146</td>
      <td>146</td>
    </tr>
    <tr>
      <th>Chinstrap</th>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
    </tr>
    <tr>
      <th>Gentoo</th>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
      <td>119</td>
    </tr>
  </tbody>
</table>
</div>



## Make a histgram plot by use plotly

A histogram is the most commonly used graph to show frequency distributions. To plot a simple histogram, we need a data frame, and a variable you want to observe.


```python
fig = px.histogram(data_frame = penguins, 
                   x = "Culmen Length (mm)", 
                   )

# show the plot
fig.show()
```

![output_12_1.png]({{ site.baseurl }}/images/output_12_1.png)

The plot looks too simple, let's add more components:

<ul>
    <li>the **color** argument takes a name of a column in data_frame and uses it to assign a color to the plot. Here, we want to assign different colors with respect to different species of penguins. Using this argument will also create a legend.</li>
    <li>the color of bars looks too bright, then we could use **opacity** argument which takes value from 0 to 1 to adjust the transparency</li>
</ul>



```python
fig = px.histogram(data_frame = penguins, 
                   x = "Culmen Length (mm)", 
                   color = "Species",
                   opacity = 0.5
                   )

# show the plot
fig.show()
```

![output_14_0.png]({{ site.baseurl }}/images/output_14_0.png)

Now, our plot looks a little better than the first version, but it may look too wide compared to its height. So by using **width** and **height** arguments, we could adjust the ratio of the plot. Also, using **nbins** can set the bins we want to show in the plot 


```python
fig = px.histogram(data_frame = penguins, 
                   x = "Culmen Length (mm)", 
                   color = "Species",
                   opacity = 0.5,
                   nbins = 30, 
                   width = 900,
                   height = 450,
                   barmode='stack'
                   )

# show the plot
fig.show()
```

![output_16_0.png]({{ site.baseurl }}/images/output_16_0.png)

Finally, add a title to the plot


```python
fig = px.histogram(data_frame = penguins, 
                   x = "Culmen Length (mm)", 
                   color = "Species",
                   opacity = 0.5,
                   nbins = 30, 
                   width = 900,
                   height = 450,
                   barmode='stack'
                   )

# set title and its alignment
fig.update_layout(title_text="Histogram of Penguins' Culmen Length", title_x=0.5, title_y=0.92)
# show the plot
fig.show()
```
![output_18_0.png]({{ site.baseurl }}/images/output_18_0.png)