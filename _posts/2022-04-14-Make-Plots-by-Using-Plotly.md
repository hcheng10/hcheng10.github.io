---
layout: post
title: Make More Plots By Using Plotly
---
To make some animation plots!

## 1. Create a Database

As always, we should import modules that we need to create a database first.
- old friend **panda**
- sqlite3: SQLite is a C library that provides a lightweight disk-based database that doesn’t require a separate server process and allows accessing the database using a nonstandard variant of the SQL query language.


```python
import pandas as pd
import sqlite3
from plotly import express as px
import numpy as np
from sklearn.linear_model import LinearRegression
```


```python
temps = pd.read_csv("temps_stacked.csv")

# rename columns
countries = pd.read_csv('countries.csv')
countries = countries.rename(columns = {"FIPS 10-4": "FIPS_10-4"}) 
countries = countries.rename(columns = {"ISO 3166": "ISO_3166"})

stations = pd.read_csv('station-metadata.csv')
```

The **'countries'** table has column names that contain space. Keeping spaces in column names in the database is not a good idea. Later when we use those columns, we need to put them within " " to specific this is a column name.

So for convenience, we could change those column names by replacing 'space' with '_'
<br>
<br>

**Steps**
<ul>
    <li>start by creating a Connection object that represents the database. Here the data will be stored in the temps.db file</li>
    <li>use to_sql() function to write records stored in a DataFrame to a SQL database</li>
    <li>Once a Connection has been established, create a Cursor object and call its execute() method to perform SQL commands:</li>
</ul>

<br>


```python
# open a connection to temps.db sp that you can 'talk' to using python
conn = sqlite3.connect("temps.db")

# if the temperatures already exits in the temps.db, if_exists="replace" could replace the cvs file.
temps.to_sql("temperatures", conn, if_exists="replace", index=False)
countries.to_sql("countries", conn, if_exists="replace", index=False)
stations.to_sql("stations", conn, if_exists="replace", index=False)

# always close your connection
conn.close()
```

<br>
<br>

Here is a simple example: 
> we want to check how many tables inside the database:

> "SELECT name FROM sqlite_master WHERE type='table'" is a sql command

> use conn.cursor() to create a Cursor object

> Cursor objects have execute() method to perform SQL commands


```python
conn = sqlite3.connect("temps.db")

# query the database
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

# retrieve the data
print(cursor.fetchall())

conn.close()
```

    [('temperatures',), ('countries',), ('stations',)]
    

From the output, we can see total 3 tables inside the database
<br>
<br>

Before, we write a function, lets check what are varibles in those tables:


```python
temps.head()
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
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>1</td>
      <td>-0.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>2</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>3</td>
      <td>4.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>4</td>
      <td>7.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>5</td>
      <td>11.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
countries.head()
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
      <th>FIPS_10-4</th>
      <th>ISO_3166</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>




```python
stations.head()
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
      <th>ID</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>STNELEV</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AE000041196</td>
      <td>25.3330</td>
      <td>55.5170</td>
      <td>34.0</td>
      <td>SHARJAH_INTER_AIRP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AEM00041184</td>
      <td>25.6170</td>
      <td>55.9330</td>
      <td>31.0</td>
      <td>RAS_AL_KHAIMAH_INTE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AEM00041194</td>
      <td>25.2550</td>
      <td>55.3640</td>
      <td>10.4</td>
      <td>DUBAI_INTL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AEM00041216</td>
      <td>24.4300</td>
      <td>54.4700</td>
      <td>3.0</td>
      <td>ABU_DHABI_BATEEN_AIR</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Write a Query Function
<br>
The return value of query_climate_database() is a Pandas dataframe of temperature readings for the specified country, in the specified date range, in the specified month of the year. This dataframe should have columns for:
<ol>   
    <li><strong>country</strong>, a string giving the name of a country (e.g. ‘United States’) for which data should be returned.</li>
    <li><strong>year_begin</strong> a integer giving the earliest years for which should be returned.</li>
    <li><strong>year_end</strong> a integer giving the latest years for which should be returned.</li> 
    <li><strong>month</strong>, an integer giving the month of the year for which should be returned.</li>
</ol>

The return value of query_climate_database() is a Pandas dataframe of temperature readings for the specified country, in the specified date range, in the specified month of the year. This dataframe should have columns for:

<ul>
    <li>The station name.</li>
    <li>The latitude of the station.</li>
    <li>The longitude of the station.</li>
    <li>The name of the country in which the station is located.</li>
    <li>The year in which the reading was taken.</li>
    <li>The month in which the reading was taken.</li>
    <li>The average temperature at the specified station during the specified year and month. (Note: the temperatures in the raw data are already averages by month, so you don’t have to do any aggregation at this stage.)</li>
</ul>

Idea: We want to open a sql connection and close it within the function call, write a sql command which takes input arguments. To achieve this task, we could use String.format(), or use panda modules.


```python
def query_climate_database(country, year_begin, year_end, month):
    # open connection to database
    conn = sqlite3.connect("temps.db")
    
    # sql command
    cmd = "SELECT S.Name, S.LATITUDE, S.LONGITUDE, C.Name AS Country, T.Year, T.Month, T.Temp \
    FROM temperatures T \
    LEFT JOIN stations S ON S.ID = T.ID \
    LEFT JOIN countries C ON SUBSTRING(S.id, 1, 2)= C.'FIPS_10-4' \
    WHERE (Year BETWEEN {} AND {}) AND (Month={})".format(year_begin, year_end, month)
    
    # excute the command
    df = pd.read_sql(cmd, conn)
    
    # subset the dataframe with the country taht user wanted
    df = df[df['Country'] == country]

    # always close the database when we are finish use it
    conn.close()
    
    return df
```


```python
df = query_climate_database(country = "India", 
                            year_begin = 1980, 
                            year_end = 2020,
                            month = 1)
```


```python
df
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>165212</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>165213</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>165214</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>165215</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>165216</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>168359</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>168360</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>168361</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>168362</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>168363</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



## 3. Write a Geographic Scatter Function for Yearly Temperature Increases
<br>

Goal: Write a function called temperature_coefficient_plot() which can make a scatter plot that indicates the average yearly change in temperature vary within a given country.

**Panda Functions:**
- transform: Call func on self producing a DataFrame with the same axis shape as self.
- apply: Apply a function along an axis of the DataFrame.

How do we find the yearly temperature increases?
- fit a line through temperatures for each station/each month, find the slope of the line and that corresponds to "average change in temp per year"

If we want to know how many observations of each stations name, and add the corresponding number of observations into the data frame form we made:


```python
df["count"] = df.groupby(["NAME"])["Country"].transform('count')
df
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>165212</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
      <td>34</td>
    </tr>
    <tr>
      <th>165213</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
      <td>34</td>
    </tr>
    <tr>
      <th>165214</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
      <td>34</td>
    </tr>
    <tr>
      <th>165215</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
      <td>34</td>
    </tr>
    <tr>
      <th>165216</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
      <td>34</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>168359</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
      <td>7</td>
    </tr>
    <tr>
      <th>168360</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
      <td>7</td>
    </tr>
    <tr>
      <th>168361</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
      <td>7</td>
    </tr>
    <tr>
      <th>168362</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
      <td>7</td>
    </tr>
    <tr>
      <th>168363</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 8 columns</p>
</div>




```python
def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs = 10, **kwargs):
    # use the query function we made to make a dataframe
    df = query_climate_database(country, year_begin, year_end, month)
    
    # use transform to add the corresponding count for each rows
    df["count"] = df.groupby(["NAME"])["Country"].transform('count')
    
    # subset the dataframe by remove the data that less than min_obs
    df = df[df['count'] >= min_obs]

    # define a function to get the linear regression for input dataframe, then return it's slope
    def coef(data_group):
        X = data_group[["Year"]]
        y = data_group["Temp"]
        LR = LinearRegression()
        LR.fit(X, y)
        slope = LR.coef_[0]
        return slope
    
    df = df.groupby(["NAME", "LATITUDE", "LONGITUDE"]).apply(coef).round(4)
    df = pd.DataFrame(df)
    df = df.reset_index()
    df = df.rename(columns = {0: "Estimated_Yearly_Increase_(℃)"})
    
    fig = px.scatter_mapbox(data_frame = df, 
                            lat = "LATITUDE", 
                            lon = "LONGITUDE",
                            hover_name = "NAME", 
                            color = "Estimated_Yearly_Increase_(℃)",
                            **kwargs
                           )
    return fig
```


```python
color_map = px.colors.diverging.RdGy_r

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map
                                  )

fig.update_layout(title="Estimates of yearly increase in temperature in January for<br>stations in India, years 1980-2020")

fig.show()
```

{% include plotly_one_hw2.html %}

```python
color_map = px.colors.diverging.RdGy_r

fig = temperature_coefficient_plot("Japan", 1980, 2020, 5, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map
                                  )

fig.update_layout(title="Estimates of yearly increase in temperature in May for<br>stations in Japan, years 1980-2020")

fig.show()
```

{% include plotly_two_hw2.html %}

## 4. Create One More Query Function and Two More Interesting Figures

### 4-1. Let's make a scatter plot with query_climate_database(): 

For this plot, we want to select specific citits of a country and output scatter plot with year verse temperature. Therefore, we could see the temperatures' change over years. With facet_col=city's name, this plot function will make multiple subplots with respect cities.

- We use query_climate_database() to make a dataframe with country (ex: China), year from 1980 to 2020, and month equal to Feburary as example.
- Use set to output unique cities' name.
- Pick cities you want to observe, and group those cities name as a list.


```python
query_climate_database("China", 1980, 2020, 2)
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108398</th>
      <td>AN_XI</td>
      <td>40.50</td>
      <td>96.0</td>
      <td>China</td>
      <td>1980</td>
      <td>2</td>
      <td>-5.27</td>
    </tr>
    <tr>
      <th>108399</th>
      <td>AN_XI</td>
      <td>40.50</td>
      <td>96.0</td>
      <td>China</td>
      <td>1981</td>
      <td>2</td>
      <td>-5.46</td>
    </tr>
    <tr>
      <th>108400</th>
      <td>AN_XI</td>
      <td>40.50</td>
      <td>96.0</td>
      <td>China</td>
      <td>1983</td>
      <td>2</td>
      <td>-5.10</td>
    </tr>
    <tr>
      <th>108401</th>
      <td>AN_XI</td>
      <td>40.50</td>
      <td>96.0</td>
      <td>China</td>
      <td>1984</td>
      <td>2</td>
      <td>-7.40</td>
    </tr>
    <tr>
      <th>108402</th>
      <td>AN_XI</td>
      <td>40.50</td>
      <td>96.0</td>
      <td>China</td>
      <td>1985</td>
      <td>2</td>
      <td>-5.90</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>123047</th>
      <td>YUANLING</td>
      <td>28.47</td>
      <td>110.4</td>
      <td>China</td>
      <td>2009</td>
      <td>2</td>
      <td>9.99</td>
    </tr>
    <tr>
      <th>123048</th>
      <td>YUANLING</td>
      <td>28.47</td>
      <td>110.4</td>
      <td>China</td>
      <td>2010</td>
      <td>2</td>
      <td>7.56</td>
    </tr>
    <tr>
      <th>123049</th>
      <td>YUANLING</td>
      <td>28.47</td>
      <td>110.4</td>
      <td>China</td>
      <td>2011</td>
      <td>2</td>
      <td>7.74</td>
    </tr>
    <tr>
      <th>123050</th>
      <td>YUANLING</td>
      <td>28.47</td>
      <td>110.4</td>
      <td>China</td>
      <td>2012</td>
      <td>2</td>
      <td>4.85</td>
    </tr>
    <tr>
      <th>123051</th>
      <td>YUANLING</td>
      <td>28.47</td>
      <td>110.4</td>
      <td>China</td>
      <td>2013</td>
      <td>2</td>
      <td>7.25</td>
    </tr>
  </tbody>
</table>
<p>14654 rows × 7 columns</p>
</div>




```python
list(set(df["NAME"]))[0:10] # just a example to use set and print first 10 cities from it.
```




    ['AGARTALA',
     'MADRAS_MINAMBAKKAM',
     'JAGDALPUR',
     'MINICOYOBSY',
     'SURAT',
     'LUDHIANA',
     'SHOLAPUR',
     'JAIPUR_SANGANER',
     'SATNA',
     'TEZPUR']



<br>Here is the function to make a scatter plot:


```python
def scatter_plot(country, year_begin, year_end, month, cities, **kwargs):
    
    df = query_climate_database(country, year_begin, year_end, month)
    
    df_new = pd.DataFrame(columns = df.columns) # make a empty dataframe
    
    # append the data that collected from input cities
    for city in cities:
        df_new = pd.concat([ df_new, df[df["NAME"]==city] ])
    
    fig = px.scatter(data_frame=df_new, 
                     x= "Year", 
                     y="Temp",
                     color = "NAME", 
                     hover_name="NAME", 
                     hover_data=["LATITUDE", "LONGITUDE"],
                     facet_col="NAME", 
                     **kwargs
                    )
    
    return fig
```


```python
cities_selected = ['HEFEI', 'GUANGZHOU', 'SHANGHAI_HONGQIAO','AN_XI']

fig = scatter_plot("China", 1980, 2020, 2, cities_selected)

fig.update_layout(title="Scatter plots with year verse temperature in Feburary with selected cities of China, years 1980-2020")

fig.show()
```

{% include plotly_three_hw2.html %}

### 4-2 Using Same Data Set To Make Another Query Function and Use It To Make a Figure

**Query Function:** 

This Query Function will take arguments country, year_begin, year_end, and their definition is same as the arguments in query_climate_database function. In this function we will have a new column 'Date' that stores data time values.


```python
def query_climate_database_new(country, year_begin, year_end):
    # open connection to database
    conn = sqlite3.connect("temps.db")
    
    # sql command
    cmd = "SELECT S.Name, C.Name AS Country, T.Year, T.Month, T.Temp \
    FROM temperatures T \
    LEFT JOIN stations S ON S.ID = T.ID \
    LEFT JOIN countries C ON SUBSTRING(S.id, 1, 2)= C.'FIPS_10-4' \
    WHERE (Year BETWEEN {} AND {}) \
    ORDER BY T.Year, T.Month".format(year_begin, year_end)
    
    # excute the command
    df = pd.read_sql(cmd, conn)
    
    # subset the dataframe with the country taht user wanted
    df = df[df['Country'] == country]
    
    # make new column with date time values
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str))

    # always close the database when we are finish use it
    conn.close()
    
    return df
```


```python
# an example
df = query_climate_database_new(country = "United States", 
                                year_begin = 2000, 
                                year_end = 2020)
```


```python
df
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
      <th>NAME</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7506</th>
      <td>ALEXANDER_CITY</td>
      <td>United States</td>
      <td>2000</td>
      <td>1</td>
      <td>7.79</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>7507</th>
      <td>ANDALUSIA_3_W</td>
      <td>United States</td>
      <td>2000</td>
      <td>1</td>
      <td>10.32</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>7508</th>
      <td>ASHLAND_3_ENE</td>
      <td>United States</td>
      <td>2000</td>
      <td>1</td>
      <td>6.89</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>7509</th>
      <td>ATHENS</td>
      <td>United States</td>
      <td>2000</td>
      <td>1</td>
      <td>5.88</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>7510</th>
      <td>ATMORE</td>
      <td>United States</td>
      <td>2000</td>
      <td>1</td>
      <td>11.24</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3416132</th>
      <td>GRAND_MARAIS</td>
      <td>United States</td>
      <td>2020</td>
      <td>12</td>
      <td>-4.29</td>
      <td>2020-12-01</td>
    </tr>
    <tr>
      <th>3416133</th>
      <td>SISSETON_MUNI_AP</td>
      <td>United States</td>
      <td>2020</td>
      <td>12</td>
      <td>-3.43</td>
      <td>2020-12-01</td>
    </tr>
    <tr>
      <th>3416134</th>
      <td>BOSCOBEL_AP</td>
      <td>United States</td>
      <td>2020</td>
      <td>12</td>
      <td>-2.62</td>
      <td>2020-12-01</td>
    </tr>
    <tr>
      <th>3416135</th>
      <td>LINCOLN_8_ENE</td>
      <td>United States</td>
      <td>2020</td>
      <td>12</td>
      <td>-1.32</td>
      <td>2020-12-01</td>
    </tr>
    <tr>
      <th>3416136</th>
      <td>LINCOLN_11_SW</td>
      <td>United States</td>
      <td>2020</td>
      <td>12</td>
      <td>-0.51</td>
      <td>2020-12-01</td>
    </tr>
  </tbody>
</table>
<p>1727789 rows × 6 columns</p>
</div>



**Line plot:**

People may interesting to observe the temperature change through the date for selected cities in a country. So we could make a line plot to achieve this goal.


```python
def line_plot(country, year_begin, year_end, cities, **kwargs):
    
    df = query_climate_database_new(country, year_begin, year_end)
    
    df_new = pd.DataFrame(columns = df.columns) # make a empty dataframe
    
    # append the data that collected from input cities
    for city in cities:
        df_new = pd.concat([ df_new, df[df["NAME"]==city] ])
    
    fig = px.line(data_frame=df_new, 
                  x= "Date", 
                  y="Temp",
                  color = "NAME", 
                  hover_name="NAME", 
                  facet_col="NAME", 
                  **kwargs
                 )
    
    return fig
```


```python
cities_selected = ['HEFEI', 'GUANGZHOU', 'SHANGHAI_HONGQIAO']

fig = line_plot("China", 2000, 2020, cities_selected)

fig.update_layout(title="Line plots with date verse temperature with selected cities of China, years 2000-2020")

fig.show()
```

{% include plotly_four_hw2.html %}