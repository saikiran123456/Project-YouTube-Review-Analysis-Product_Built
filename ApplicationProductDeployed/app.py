# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:06:22 2022

@author: AbhiSai
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from PIL import Image
st.set_page_config(
   page_title="YouTube Review Analysis",
   page_icon='logo.png',
)
def run():
    img = Image.open('youtube.png')
    # img = img.resize((250,250))
    st.image(img)
    st.markdown('''<h2 style='text-align: left; color: #ff3300;'> Channel & Review Analysis Dashboard:</h5>''', unsafe_allow_html=True)
    link = 'Data Scientist at Excelr Solutions, Client Name: A.I Variant. You can reach me '    '[©Developed by Saikiran Dasari](https://www.linkedin.com/in/saikiran-dasari-265718a8/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    
run()


st.write("---")

# Youtube Channel videos Scecific Scrapping using A) Beautiful soup and B) Youtube Data V3 API-Key from Google-Cloud Console

# 1) Channel Analysis:
st.markdown('''<h2 style='text-align: left; color: #ffffff;'> YouTube Channel Analysis:</h5>''', unsafe_allow_html=True)

# A) Importing the BeautifulSoup libraries for Retrieving Channel Data using YouTube video Link
from bs4 import BeautifulSoup as soup
import requests
import re
import json

if 'status' not in st.session_state:
    st.session_state['status'] = 'submitted'

def refresh_state():
	st.session_state['status'] = 'submitted'

# BeautifulSoup
link = st.write("You can Go ahead and Enable 'YouTube Data API v3' from Google Cloud Console and Get 'API-Key' for yourself here: ",'https://console.cloud.google.com/apis/library/youtube.googleapis.com?project=bigquery-sandbox-330718', '\n','(NOTE: You need to login using your gmail account)')

api_key = st.text_input("Input your API-Key 👇:", value="", type="password",on_change=refresh_state)

my_url = st.text_input('Enter your YouTube video link', on_change=refresh_state, type = 'default')

button = st.button('ENTER')
import time
time.sleep(1)




if my_url is not None:
    with st.spinner('Loading the channel analysis...'):
        time.sleep(1)
    st.text("Please wait...,If there is any error just try Input your key & link above & hit ENTER")
    st.video(my_url)

st.text("Your API-Key is " + st.session_state['status'])
st.text("Your Video is " + st.session_state['status'])


import time
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.001)
    my_bar.progress(percent_complete + 1)
button = st.button('Scrapped: Title, Total_Video_Views, Total_Video_Likes, Video_Description, Channel_id, Video_id')

# Setting up beautiful json parser 
s=soup(requests.get(my_url,cookies={'CONSENT': 'YES+1'}).text,'html.parser')
data=re.search(r"var ytInitialData = ({.*});", str(s.prettify())).group(1)
json_data=json.loads(data)

# Channel Name Extraction:
path=json_data['contents']['twoColumnWatchNextResults']['results']['results']['contents']
final_channel_name = path[1]['videoSecondaryInfoRenderer']['owner']['videoOwnerRenderer']['title']['runs'][0]['text'] #['title']['runs'][0]['text']
st.write('*Channel Name is:*                   ', '\n', final_channel_name)


#Title of the YouTube Video
path=json_data['contents']['twoColumnWatchNextResults']['results']['results']['contents']
video_title=path[0]['videoPrimaryInfoRenderer']['title']['runs'][0]['text']
st.write('*Title :*                ','\n',video_title)

#Selected Video Views
selected_video_views=path[0]['videoPrimaryInfoRenderer']['viewCount']['videoViewCountRenderer']['viewCount']['simpleText']
selected_video_views=selected_video_views.split(' ')[0]
selected_video_views=selected_video_views.replace(',','')
selected_video_views=int(selected_video_views)
st.write('*Selected Video Views :*            ','\n',selected_video_views)


# Extracting Selected Video Likes!
cid=json_data['contents']['twoColumnWatchNextResults']['results']['results']['contents']
clogo=cid[1]['videoSecondaryInfoRenderer']['owner']['videoOwnerRenderer']['thumbnail']['thumbnails'][2]['url']

selected_video_likes=cid[0]['videoPrimaryInfoRenderer']['videoActions']['menuRenderer']['topLevelButtons'][0]['segmentedLikeDislikeButtonRenderer']['likeButton']['toggleButtonRenderer']['defaultText']['accessibility']['accessibilityData']['label']
selected_video_likes = selected_video_likes.replace(',', '')
selected_video_likes = selected_video_likes.replace('.', '')
st.write('*Selected Video Likes :*           ' , '\n',selected_video_likes)

#video_desc=path[1]['videoSecondaryInfoRenderer']['description']['runs'][0]['text']
#st.write('Video: Description      ','\n', video_desc)

# Extracting Channel_id
final_channel_id=path[1]['videoSecondaryInfoRenderer']['owner']['videoOwnerRenderer']['title']['runs'][0]['navigationEndpoint']['browseEndpoint']['browseId']
st.write('*Channel id  is :*           ','\n', final_channel_id)

# Extracting Video_id
h=json_data['currentVideoEndpoint']
videoid=h['watchEndpoint']['videoId']
st.write('*Video_id in the link is :*      ', '\n',videoid)



# B) Google API-KEY YouTube v3 API Service: ----------------
from googleapiclient.discovery import build

youtube = build('youtube', 'v3', developerKey=api_key)
channel_ids=[final_channel_id]

# Function to Scrape the Data
def get_channel_stats(youtube, channel_ids):
    all_data = []
    request = youtube.channels().list(
                part='snippet,contentDetails,statistics',
                id=','.join(channel_ids))
    response = request.execute() 
    
    for i in range(len(response['items'])):
        data = dict(Channel_Name = response['items'][i]['snippet']['title'],
                    Subscribers = response['items'][i]['statistics']['subscriberCount'],
                    Published_At = response['items'][i]['snippet']['publishedAt'],
#                    Country = response['items'][i]['snippet']['country'],
                    Views = response['items'][i]['statistics']['viewCount'],
                    Total_videos = response['items'][i]['statistics']['videoCount'],
                    playlist_id = response['items'][i]['contentDetails']['relatedPlaylists']['uploads'],
                    Description = response['items'][i]['snippet']['description'])
        all_data.append(data)
    
    return all_data
# Scrapped into DataFrame the: Channel_Name, Subscribers, Views, Total_videos, playlist_id
channel_statistics = get_channel_stats(youtube, channel_ids)
channel_data = pd.DataFrame(channel_statistics)


# Additional Data Extraction and Merging with the channel_data dataframe
channel_data['selected_video_likes']=selected_video_likes
channel_data['selected_video_views']=selected_video_views
channel_data['videoid']=videoid
channel_data['video_title']=video_title


# Assigning the correct data types for the required variables
# Removing the string 'likes' from 57710 likes (selected_video_likes column)
a = []
line = selected_video_likes
for word in line.split():
    try:
        a.append(float(word))
    except ValueError:
        pass
selected_video_likes=int(a[0])
channel_data['selected_video_likes']=selected_video_likes


channel_data['Subscribers'] = pd.to_numeric(channel_data['Subscribers'])
channel_data['Published_At'] = pd.to_datetime(channel_data['Published_At']).dt.date
channel_data['Views'] = pd.to_numeric(channel_data['Views'])
channel_data['Total_videos'] = pd.to_numeric(channel_data['Total_videos'])

# Rounding the Values
channel_data['Views']=np.round(channel_data['Views'])
channel_data['Subscribers']=np.round(channel_data['Subscribers'])
channel_data['selected_video_views']=np.round(channel_data['selected_video_views'])

st.text('Here, We get: Channel_Name, Subscribers, Published_At, Country, Entire_Channel_Video_ViewCount, Total_videos, playlist_id, Channel Description, Selected_Video_Likes, Selected_Video_Views, Video_ID, Video_Title')
#button = st.button('The Scrapped: Title, Total_Video_Views, Total_Video_Likes, Video_Description, Channel_id, Video_id')

cdata = channel_data.copy()
st.dataframe(cdata)

st.write("---")

    

    

# 2) YouTube Video Specific Comments Analysis----------------

st.subheader('YouTube Comments Analysis:')

videoid = st.text_input('Input Video ID from above or your required one!! ') 
button = st.button('enter')
st.text("Please wait...,If there is any error just try Input your YouTube Video_id above")

ID = videoid
box = [['Name', 'Comment', 'Time', 'Likes', 'ReplyCount']]

# --------------------------------------------

# Created a Module{myScrapingFunction.py} and UserDefinedFunction{scrape_comments_with_replies} locally to Scrape the Data and Custom Importing the same!!
from myScrapingFunction import scrape_comments_with_replies

# Scrapping the required data!
video_data = scrape_comments_with_replies(ID, box, youtube)
# TimeStamp Noted:  6:14pm to 6:16pm (Only took 2 Minutes to Scrape 4710 Sized data!!)

import time
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.001)
    my_bar.progress(percent_complete + 1)
button = st.button('Scrapped and Feature Engineered the Data!!!')
#st.dataframe(video_data)


def cdata1():
    cdata = video_data.copy()
    cdata = cdata.iloc[:,:].values
    cdata = pd.DataFrame(cdata,columns=['Name','Comment','Time','Likes','ReplyCount'])
    cdata = cdata.drop(0)
    cdata['ReplyCount']=cdata['ReplyCount'].dropna()
    cdata['ReplyCount']=cdata['ReplyCount'].replace('' ,0)
    cdata['ReplyCount'] = pd.to_numeric(cdata['ReplyCount'])
    cdata['Time'].dropna(inplace=True) 
    cdata['Time'] =  pd.to_datetime(cdata["Time"], format="%Y-%m-%dT%H:%M:%S")
    cdata['date'] = [d.date() for d in cdata['Time']]
    cdata['time'] = [d.time() for d in cdata['Time']]      
    cdata.drop('Time',axis=1)
    
    cdata['Month'] = pd.to_datetime(cdata['date']).dt.strftime('%b')
    cdata['Year'] = pd.to_datetime(cdata['date']).dt.strftime('%Y')
    cdata['Day'] = pd.to_datetime(cdata['date']).dt.strftime('%d')
    
    
    cdata['hour'] = pd.to_datetime(cdata['time'], format='%H:%M:%S').dt.hour
    b = [0,4,8,12,16,20,24]
    l = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
    cdata['session'] = pd.cut(cdata['hour'], bins=b, labels=l, include_lowest=True)

    def f(x):
        if (x > 4) and (x <= 8):
            return 'Early Morning'
        elif (x > 8) and (x <= 12 ):
            return 'Morning'
        elif (x > 12) and (x <= 16):
            return'Noon'
        elif (x > 16) and (x <= 20) :
            return 'Eve'
        elif (x > 20) and (x <= 24):
            return'Night'
        elif (x <= 4):
            return'Late Night'
    cdata['session'] = cdata['hour'].apply(f)

    return cdata


ddata = cdata1()
# Backing Up the ddata into NewData!!
NewData = ddata.copy()  #copying the above processed data in a new object called NewData


st.dataframe(NewData)
st.write('Shape of your dataset is:',NewData.shape)



# --------------- Visualizations -------------------------------------

st.header('VISUALIZATIONS:')

st.set_option('deprecation.showPyplotGlobalUse', False) #Streamlit Warning 

# 1) Sorting Top 5 People whose comments got maximum Reply_Counts!!:
top5_ReplyCount_users = NewData.sort_values(by='ReplyCount', ascending=False).head(5)
st.write('----------People who Got Max Reply Count on their Comments----------')
plt.title('Number of Reply Counts per Person')
sns.set(rc={'figure.figsize':(10,6)})
sns.barplot(y = 'Name', x = 'ReplyCount', data=top5_ReplyCount_users)
plt.show()
st.pyplot()
st.write(top5_ReplyCount_users[['Name','ReplyCount']].head(1),'is at Top position who got Maximum Reply Comments')

st.write("---")

# 2) Top People who got Maximum Likes on their comments
st.write('----------Number of Likes Count per Person:----------')
top5_Likes_users = NewData.sort_values(by='Likes', ascending=False).head(5)
plt.title('Number of Likes Count per Person')
sns.set(rc={'figure.figsize':(15,15)})
sns.barplot(x = 'Name', y='Likes', data= top5_Likes_users)
plt.show()
st.pyplot()
st.write(top5_Likes_users[['Name','Likes']].head(1),'is at Top position who got Maximum Likes on their Comments')

st.write("---")

st.set_option('deprecation.showPyplotGlobalUse', False)    # This resolved the error/warrning in streamlit 

# 3) Comments per Month:
Comments_per_month = NewData.groupby('Month', as_index=False).size()
# Using Categorical Index
sort_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Comments_per_month.index = pd.CategoricalIndex(Comments_per_month['Month'], categories=sort_order, ordered=True)
st.write('----------Comments_per_month Bar chart:----------')
st.bar_chart(data=Comments_per_month,x='Month', y='size', width=0, height=0, use_container_width=True)


st.write("---")

# -------------- BOKEH Visualizations --------------------------------------------
# from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6

# 4) Comments per Year:
Comments_per_year =  NewData.groupby('Year', as_index=False).size()
Year = Comments_per_year['Year']
counts = Comments_per_year['size']
source = ColumnDataSource(data=dict(Year=Year, counts=counts, color=Spectral6))
st.write('----------Total Comments in each Year----------')
p = figure(x_range=Year, height=400,  title="----------Comments_per_Year Bar chart:----------")
p.xaxis.axis_label = "YEARS"
p.yaxis.axis_label = "Total Comments in each Year"
p.vbar(x='Year', top='counts', width=0.6, color='color', legend_field="Year", source=source)
p.legend.orientation = "horizontal"
p.legend.location = "top_center"
st.bokeh_chart(p, use_container_width=True)


st.write("---")


# 5) Comments per Day:
Comments_per_day = NewData.groupby('Day', as_index=False).size()
st.write('----------Comments_per_Day line chart----------')

x = Comments_per_day['Day']
y = Comments_per_day['size']

p = figure(height=400,
    title='simple line example',
    x_axis_label='Each Days on X-axis',
    y_axis_label='Total Comments in each day')
p.line(x, y, legend_label='Trend', line_width=2)
st.bokeh_chart(p, use_container_width=True)

st.write("---")

# 6)Session wise comments
st.write('----------Session wise comments-----------------')
sns.catplot(data = NewData, x = 'session', kind='count', palette='ch: 25',height=7, aspect=2.0)
plt.show()
st.pyplot()

st.write("---")

# 7) Cat Plot according to Session wrt Month
st.write('----------Interaction of Comments session wrt Month----------')
sns.catplot(
    data=NewData, x="Month", hue="session", kind="count",
    palette="pastel", edgecolor=".6", aspect=11.7/8.27
)
plt.show()
st.pyplot()

st.write("---")

# 8)  Likes Vs Year
sns.stripplot(y='Likes',x='Year',data=NewData)
st.write('----------Year wise likes based on comments----------')
plt.title("Year wise likes based on comments", fontsize=10);
plt.show()
st.pyplot()

st.write("---")

# 9) Session wise likes based on Comments
data = NewData.groupby("session")["Likes"].sum()
pie, ax = plt.subplots(figsize=[10,6])
labels = data.keys()
plt.pie(x=data, autopct="%.1f%%",labels=labels, pctdistance=0.5)
st.write('----------Session wise likes wrt Comments----------')
plt.title("Session wise likes wrt Comments", fontsize=10);
plt.show()
st.pyplot()

st.write("---")


# =============================================================================

st.title('Natural Language Processing')

#  ------------Comments Column Cleaning-------------

# Importing spacy, nltk, nltk.corpus(stopwords), 
import spacy # language models
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import contractions
import inflect
from wordcloud import WordCloud
from nltk import word_tokenize

    
nlp = spacy.load("en_core_web_md")
        

def number_to_text(data):
    temp_str = data.split()
    string = []
    for i in temp_str:

    # if the word is digit, converted to
    # word else the sequence continues

        if i.isdigit():
            temp = inflect.engine().number_to_words(i)
            string.append(temp)
        else:
            string.append(i)
    outputStr = " ".join(string)
    return outputStr


stop_words = stopwords.words('english')
lemma = WordNetLemmatizer()
def lemmatiz_text(data):    
    tokens = word_tokenize(data)
    lemma_tokens = [lemma.lemmatize(word, pos='v') for word in tokens if word not in (stop_words)]
    return " ".join(lemma_tokens)


def cleantext(text):
    
    text = re.sub(r'[^\w\s]', " ", text) # Remove punctuations
    
    text = re.sub(r"https?:\/\/\S+", ",", text) # Remove The Hyper Link
    
    text = contractions.fix(text) # remove contractions 
    
    text = number_to_text(text) # convert numbers to text   
    
    text = text.lower() # convert to lower case
        
    text = lemmatiz_text(text) # lemmatization
    
    return text


# Backing up the NewData!!
cleandata = NewData.copy()
cleandata["clean_text"] = cleandata["Comment"].apply(cleantext)

import time
my_bar1 = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.001)
    my_bar1.progress(percent_complete + 1)
button2 = st.button('Comments Clean Text Column Data!')
st.dataframe(cleandata)



# ------------------ NLP Visualizations -------------------------------------------------

st.header('NLP Visualizations:')

complete_review_string = " ".join ([rev for rev in cleandata["clean_text"]])
words = nltk.word_tokenize(complete_review_string)

# 1) Based on the Highest Frequency Word Counts from entire Video Comments
st.write('Line Chart showing Highest Frequency Word Counts from entire Video Comments: ')
from nltk.probability import FreqDist
fdist = FreqDist(words)
fdist.plot(20,cumulative=False)
plt.show()
st.pyplot()

st.write("---")

# 2) Wordcloud based on Frequencies irrespective of Positive words/Negative Words
st.write('Word Cloud based on Highest Frequencies irrespective of Positive/Negative words: ')
wordCloud = WordCloud(width = 1000, height = 700, random_state = 21, max_font_size = 119).generate(complete_review_string)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()
st.pyplot()

st.write("---")

# 3) Importing POSITIVE word corpus and visualizing and Checking the Frequencies
with open("positive-words.txt","r",  encoding='utf-8') as pos:
    poswords = pos.read().split("\n")
review_pos = " ".join ([w for w in words if w in poswords])

# Positive Word Cloud
wordCloudPos = WordCloud(width = 1000, height = 700, random_state = 21, max_font_size = 100).generate(review_pos)
st.write('Word Cloud Plot based on the Positive Frequency tokens!')
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloudPos, interpolation = "bilinear")
plt.axis("off")
plt.show()
st.pyplot()


# Analyzing the number of Positive Words in the Reviews
pos_words = nltk.word_tokenize(review_pos)
fdistPos = FreqDist(pos_words)

TotalPositiveWords = len(pos_words)
st.write('*Top 10 Positive word tokens with their counts*:', '\n')
st.write(TotalPositiveWords)

pos_words = nltk.word_tokenize(review_pos)
fdistPos = FreqDist(pos_words)
st.write(fdistPos.most_common(10))



st.write("---")



# Frequence Distribution Bar Chart Visuals (applied clean_text function and visualize)
stop_words = stopwords.words('english') # remove stop words

# Backing up the cleandata!!
cleandata1 = cleandata.copy()

cleandata1["clean_text1"] = cleandata["clean_text"].apply(cleantext)
st.write('Bar Chart showing Top 5 Frequency tokens: ')
freq = pd.Series(' '.join(cleandata1['clean_text1']).split()).value_counts()[:5]
freq = pd.DataFrame(data=freq)
freq.columns =['Frequency']
freq.reset_index(inplace = True)
freq.rename(columns = {'index':'words'},inplace = True)
plt.figure(figsize=(11,7))
sns.barplot(x='words',y='Frequency', data=freq)
plt.title('Frequency count of common word:')
plt.show()
st.pyplot()


st.write("---")


st.subheader('Final Sentiment Score of the Entire Reviews: ')
# Sentiment Analysis using TextBLOB (Introduced New Variable 'Sentiment_score')
from textblob import TextBlob
cleandata1['sentiment Score'] = cleandata['clean_text'].apply(lambda x: TextBlob(x).sentiment[0] )
# New Variable 'Sentiment: having Positive, Neutral, Negative'
cleandata1['Sentiment'] = cleandata1['sentiment Score'].apply(lambda s : 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))

# LabelEncoder for Endoding the Sentiment Categorical Column!! into (Negative,Neutral,Positive) as (0,1,2)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cleandata1['Sentiment'] = le.fit_transform(cleandata1['Sentiment'])

plt.figure(figsize=(10,7))
sns.countplot( x='Sentiment', y=None, data = cleandata1)
plt.title('2 is Positive, 1 is Neutral and 0 is Negative (distribution of Sentimental Scores)')
plt.show()
st.pyplot()


st.write("---")

# Importing NEGATIVE word corpus and visualizing and Checking the Frequencies
with open("negative-words.txt","r", encoding='utf-8') as pos:
   negwords = pos.read().split("\n")  
 
review_neg = " ".join ([w for w in words if w in negwords])


# Analyzing the number of Negative Words in the Reviews
neg_words = nltk.word_tokenize(review_neg)
fdistNeg = FreqDist(neg_words)

TotalNegativeWords = len(neg_words)
st.write('*Top 10 Negative word tokens with their counts:*', '\n')
st.write(TotalNegativeWords)

neg_words = nltk.word_tokenize(review_neg)
fdistNeg = FreqDist(neg_words)
st.write(fdistNeg.most_common(10))


# NEGATIVE Word Cloud
wordCloudNeg = WordCloud(width = 1000, height = 700, random_state = 21, max_font_size = 119).generate(review_neg)
st.write('Word Cloud Plot based on the Negative Frequency!')
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloudNeg, interpolation = "bilinear")
plt.axis("off")
plt.show()
st.pyplot()



st.stop()



# ----------------------------------END-----------------------------------










