---
layout: post
title: How to Make Histogram with Plotly
---
What’s your favorite movie or TV show?

## Intro to Web Scraping

In this post, we will find out What movies or TV shows share actors with our favorite movie or show by using Scrapy. The website I will scrape in this post is IMBD,  and explain the steps on how to scrape data from the web.

**steps:**
1) Locate the Starting IMDB Page
    -  Pick a favorite movie or TV show, and locate its IMDB page. For example, my favorite film is 'Spider-Man: No Way Home'. Its IMDB page is at: https://www.imdb.com/title/tt10872600/
2) Dry-Run Navigation:
    - First, click on the All Cast & Crew link. This will take us to a page with URL of the form '<original_url>fullcredits/'
    - Next, scroll until you see the Series Cast section. Click on the portrait of one of the actors. This will take us to a page with a different-looking URL. For example, the URL for Tom Holland is: https://www.imdb.com/name/nm4043618/?ref_=ttfc_fc_cl_t1
    - Finally, scroll down until we see the actor’s Filmography section. Note the titles of a few movies and TV shows in this section.
3) Initialize The Project
    - Our scraper is going to replicate step 2). Starting with the 'Spider-Man: No Way Home', it’s going to look at all the actors in that movie, and then log all the other movies or TV shows that they worked on.
4) Tweak Settings
    - add *CLOSESPIDER_PAGECOUNT = 20* in the settings.py file to prevents our scraper from downloading too much data while we’re still testing things out. We’ll remove this line later.

## Start with Scrapy

Before we starting the project, you may ask what is Scrapy. Briefy talk, Scrapy is a web-crawling framework written in Python. In order to use it, we need install it into our python environment (link for installation guide: https://docs.scrapy.org/en/latest/intro/install.html#intro-install). Once installed, we can start our first Scrapy project. <br>
- Open the terminal, and enter a directory where we’d like to store your code and run: <br> 
  *scrapy startproject IMBD_scraper* <br>
  This will create a IMBD_scraper folder with a lot of files inside, we dont need modfile most of them.
- Spiders are classes that you define and that Scrapy uses to scrape information from a website (or a group of websites). <br>
  They must subclass Spider and define the initial requests to make, optionally how to follow links in the pages, and how to parse the downloaded page content to extract data.
- I will name the code file as **imbd_spider.py** and save it under the **IMBD_scraper/spiders** directory in the project:<br>
  This is the code for **imbd_spider.py**
<br>
<br>

### overall of the imbd_spider.py
The details of the member functions explanation are followed by this code.


```python
import scrapy

class ImdbSpider(scrapy.Spider):
    name = 'imdb_spider'
    
    start_urls = ['https://www.imdb.com/title/tt10872600/'] # start url

    def parse(self, response):
        next_page = response.css("a[href*='fullcredits']").attrib['href']  # get element that contain url

        if next_page: # identical to if next_page is not None
            next_page = response.urljoin(next_page) # extract the url
            yield scrapy.Request(next_page, callback = self.parse_full_credits) # call parse_full_credits


    def parse_full_credits(self, response):
        cast_link = [a.attrib["href"] for a in response.css("td.primary_photo a")]

        if cast_link:
            for c in cast_link:
                next_page = response.urljoin(c)
                yield scrapy.Request(next_page, callback = self.parse_actor_page) # call parse_actor_page


    def parse_actor_page(self, response):
        actor_name = response.css(".header").css("span.itemprop::text").get()

        for moive in response.css("div.filmo-category-section:not([style*='display:none;']) b"):
            yield {
                "actor" : actor_name, 
                "movie_or_TV_name" : moive.css("a::text").get()
            }
```

As we can see, there is Spider subclasses <u>scrapy.Spider</u> and defines some attributes and methods:
- **name**: identifies the Spider. It must be unique within a project, that is, you can’t set the same name for different Spiders.
- **start_urls**: must return an iterable of Requests (you can return a list of requests or write a generator function) which the Spider will begin to crawl from. Here I used https://www.imdb.com/title/tt10872600/ (Spider-Man: No Way Home).
- **parse()**, **parse_full_credits()**, and **parse_actor_page()**: methods that will be called to handle the response downloaded for each of the requests made. Here I have 3 prase() methods, I will explain those methods in details later.<br>
  The parse() method usually parses the response, extracting the scraped data as dicts and also finding new URLs to follow and creating new requests (Request) from them.
- **yield**: inbuilt way of saving and storing data

### parse(self, response)


```python
def parse(self, response):
    next_page = response.css("a[href*='fullcredits']").attrib['href']  # get element that contain url

    if next_page: # identical to if next_page is not None
        next_page = response.urljoin(next_page) # extract the url
        yield scrapy.Request(next_page, callback = self.parse_full_credits) # call parse_full_credits
```

the start_url is link to this page, and we want jump to **Cast & crew** by defining the parse() method:<br>
<figure>
    <img src="{{ site.baseurl }}/images/hw2_pic/hw2_p1.png" alt="image missing!" style="width: 600px;"/>
    <figcaption>Start_urls: https://www.imdb.com/title/tt10872600/</figcaption>
</figure>

The parse(self, response) method will do the same step automatically when we call it. So I want this method to crawl the URL of the cast & crew page. The IMBD websites are written in HTML, we could use  CSS selectors to specify which content we want to crawl from the web. 
This is copy of the HTML that contain URL of the cast & crew page in the website: 
<a href="fullcredits/?ref_=tt_ql_cl" class="ipc-link ipc-link--baseAlt ipc-link--inherit-color">Cast &amp; crew</a>`
- **a[href*='fullcredits']** selector will select every \<a\> element whose href attribute value contains the substring "fullcredits",
- by calling **response.css("a[href*='fullcredits']").attrib['href']**, we got **'fullcredits/?ref_=tt_ql_cl'** returned in this case, but it is just a hyperlink,
- scrapy has a build in method to accesss the hyperlink: **response.urljoin(h)** where h is the hyperlink, **response.urljoin('fullcredits/?ref_=tt_ql_cl')** returns **'https://www.imdb.com/title/tt10872600/fullcredits/?ref_=tt_ql_cl'** in this case,
- then the last step of parse(self, response) is to access this link by calling **parse_full_credits()** method.

### parse_full_credits(self, response)


```python
def parse_full_credits(self, response):
    cast_link = [a.attrib["href"] for a in response.css("td.primary_photo a")]

    if cast_link:
        for c in cast_link:
            next_page = response.urljoin(c)
            yield scrapy.Request(next_page, callback = self.parse_actor_page) # call parse_actor_page
```

click the **Cast & crew**, it navigate to the Cast & Crew page: <br>
<figure>
    <img src="{{ site.baseurl }}/images/hw2_pic/hw2_p2.png" alt="image missing!" style="width: 600px;"/>
    <figcaption>cast & crew page</figcaption>
</figure>

On this page, we can find a completed cast(in credits order) listed. We want to define **parse_full_credits(self, response)** to access the all actors' page. 
From the html, we find <td class="primary_photo">...<\td> defines a standard data cell that contain the primary photos of each actors in the HTML table
Same idea as the **prase(self, response)** method, but we need use a for loop to access each actor's personal website. 
- response.css("td.primary_photo a") select all \<td\> tag with class="primary_photo", then select select all \<a\> tag within.
- for a in response.css("td.primary_photo a") access the elements iteratively
- a.attrib["href"] returns the https link
- for each link, we calling **parse_actor_page(self, response)** to access the acctor's personal web
- in this example, the first iteration will access Tom Holland's IDMB personal page, the second iteration will access Zendaya's page, and so on.

### parse_actor_page(self, response)

Finally, we can start to introduce the parse_actor_page(self, response) method


```python
def parse_actor_page(self, response):
        actor_name = response.css(".header").css("span.itemprop::text").get()

        for moive in response.css("div.filmo-category-section:not([style*='display:none;']) b"):
            yield {
                "actor" : actor_name, 
                "movie_or_TV_name" : moive.css("a::text").get()
            }
```

For example, click the **Tom Holland**, it navigate to the this page. We scroll down to the filmography section <br>
<figure>
    <img src="{{ site.baseurl }}/images/hw2_pic/hw2_p3.png" alt="image missing!" style="width: 600px;"/>
    <figcaption>filmography</figcaption>
</figure>

On this page, we can see the name of each movie or TV show that Tom Holland participate in.

Inside **parse_actor_page(self, response)** method:
- response.css(".header").css("span.itemprop::text").get() returns plain text: Tom Holland
- response.css("div.filmo-category-section:not([style*='display:none;']) b") access all the movie or TV shows
- In for loop we use yield to storing data: <u>actor's name</u> and <u>movie or TV shows</u> 

<br>

## Run Our Spider

Now, we have our imbd_spider.py setup. To use it we just type **scrapy crawl imdb_spider -o results.csv** in the terminal. The **-o** tells the program to save the data into the file that user-specified. Here, we saved data into the results.csv file. Let's see what we have in the results.csv 


```python
import pandas as pd

df = pd.read_csv("IMDB_scraper/results.csv")
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
      <th>actor</th>
      <th>movie_or_TV_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zany Dunlap</td>
      <td>Spider-Man: No Way Home</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Andrew Dunlap</td>
      <td>Spider-Man: No Way Home</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kathleen Cardoso</td>
      <td>Spider-Man: No Way Home</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kathleen Cardoso</td>
      <td>Out with the Old</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kathleen Cardoso</td>
      <td>Vengeance: Killer Lovers</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>Alfred Molina</td>
      <td>Indiana Jones and the Raiders of the Lost Ark</td>
    </tr>
    <tr>
      <th>3017</th>
      <td>Alfred Molina</td>
      <td>Bognor</td>
    </tr>
    <tr>
      <th>3018</th>
      <td>Alfred Molina</td>
      <td>A Nightingale Sang in Berkeley Square</td>
    </tr>
    <tr>
      <th>3019</th>
      <td>Alfred Molina</td>
      <td>The Song of the Shirt</td>
    </tr>
    <tr>
      <th>3020</th>
      <td>Alfred Molina</td>
      <td>The Losers</td>
    </tr>
  </tbody>
</table>
<p>3021 rows × 2 columns</p>
</div>




```python
df = df.groupby("movie_or_TV_name").count()
df = df.reset_index()
df = df.rename(columns = {"actor": "number of shared actors"})
df = df.sort_values(by="number of shared actors", ignore_index=True, ascending=False)
```


```python
df[:20] # lists top 20 movies or tv shows by the number of shared actors
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
      <th>movie_or_TV_name</th>
      <th>number of shared actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Spider-Man: No Way Home</td>
      <td>87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spider-Man: Far from Home</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spider-Man: Homecoming</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Doom Patrol</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Avengers: Endgame</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Black Lightning</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The Tomorrow War</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Avengers: Infinity War</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Richard Jewell</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Spider-Man 3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>The Daily Bugle</td>
      <td>6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Watchmen</td>
      <td>6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Heels</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spider-Man 2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>The Simpsons</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Homicide Hunter</td>
      <td>5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Dynasty</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Spider-Man</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>The Resident</td>
      <td>5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ozark</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns

df = df[:20]
ax = sns.barplot(x="movie_or_TV_name", y="number of shared actors", data=df)
ax.tick_params(axis='x', rotation=90)
for i in ax.containers:
    ax.bar_label(i,)
ax.set(xlabel='moives or TV-shows')
ax.set(title='Bar Plot of Moives or TV-Shows Verse Shared Actors')
```




    [Text(0.5, 1.0, 'Bar Plot of Moives or TV-Shows Verse Shared Actors')]




    
![png]({{ site.baseurl }}/images/output_24_1.png)
    


As I expected, the moives Spider-Man 3, Spider-Man: Far From Home, and Avengers are on the top 20s. If you like the No Way Home, you may also like the other moives on the top 20s.
<br>
GitHub repository: 
