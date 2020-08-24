from flask import Flask, render_template, request, session, url_for, redirect, flash
from flask_mysqldb import MySQL
import re
import MySQLdb
import csv
import os
import matplotlib.pyplot as plt
import tweepy
from flask import Flask, render_template, request
import nltk
nltk.download('wordnet')
import string
import pandas as pd
import numpy as np
from langdetect import detect
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from pandas.plotting import table
from FollowersAnalysis import Fetch_Follower_List
from newdatabase1 import Followers1,Result,Base
from friendsdatabase import Friends,Result,Base
from FriendsAnalysis import Fetch_Freinds_List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tweepy import OAuthHandler, API,TweepError
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.pool import StaticPool
from datetime import datetime, date, time, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def getExceptionMessage(msg):
    words = msg.split(' ')
    errorMsg = ""
    for index, word in enumerate(words):
        if index not in [0,1,2]:
            errorMsg = errorMsg + ' ' + word
    errorMsg = errorMsg.rstrip("\'}]")
    errorMsg = errorMsg.lstrip(" \'")
    return errorMsg


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def preprocess(data):
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'if', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y']
    check = Speller(lang='en')
    lemmatizer = WordNetLemmatizer()
    data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))#convert the entire string to lowercase
    data['text'] = np.vectorize(remove_pattern)(data['text'], "@[\w]*") #remove the userhandles
    data['text'] = np.vectorize(remove_pattern)(data['text'], "#[\w]*") #remove the hashtags
    data['text'] = data['text'].str.replace("[^a-zA-Z]", " ") #replace anything other than character with '' to remove b Rt,1..
    data['text'] = data['text'].str.replace('https?:\/\/.*[\r\n]*','') #remove urls
    data['text'] = data['text'].str.replace('http','')#remove http
    data['text'] = data['text'].str.replace('[^\w\s]','') # replace whitespaces or spaces with ''
    data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) # remove stopwords
    data['text'] = data['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2])) # remove all words having length<3
    for i in range(0, len(data)):
        data['text'][i] = check(data['text'][i]) #perform spelling corrections
    data['text']=data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])) #lemmatization
    return data


def model(tweet):
    data = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\FinalProject\PreprocessedDataset3.csv")
    data.dropna(inplace=True)
    x = data['text']
    y = data['label']
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 2))
    x_tfidf = tfidf_vect.fit_transform(x)
    tweet_tfidf = tfidf_vect.transform(tweet)
    logistics = LogisticRegression()
    logistics.fit(x_tfidf, y)
    predicted_value = logistics.predict(tweet_tfidf)
    print(predicted_value)
    return predicted_value


def api_creation():
    access_key = "1136284891031597056-YLsUhgdABB6i7h4TSLOWvJFs7RlXxs"
    access_secret = "QKXYm4F2j8l9jbxUz9jLPVnWVcJAxs7WQTeYPGOa7Wq5v"
    consumer_key = "hcoh0UTyaBrbCDP1QjKriOjXc"
    consumer_secret = "Jtw3TB8CmyzUTlHkt4UHhvg3zcM1KMm4RBM9N2ebBxOedgnx8o"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    return api


def get_tweets(username):
    api = api_creation()
    number_of_tweets = 100
    tweets_for_csv = []
    print('collecting tweets')
    for tweet in tweepy.Cursor(api.user_timeline, screen_name = username).items(number_of_tweets):
        tweets_for_csv.append([tweet.created_at, tweet.text.encode("utf-8"),  tweet.user.profile_image_url, tweet.user.followers_count, tweet.user.friends_count])
    outfile = r"C:\Users\LENOVO\PycharmProjects\FinalProject\account_tweets.csv"
    with open(outfile, 'w+', encoding = "utf-8", newline = "") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['created_at', 'text', 'profile_img', 'followers', 'following'])
        writer.writerows(tweets_for_csv)
    import pandas as pd
    read_tweets = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\FinalProject\account_tweets.csv")
    return read_tweets


def plot_pie(result):
    bully = 0
    notbully = 0
    for i in range(0, len(result)):
        if (result[i] == 0):
            notbully += 1
        else:
            bully += 1
    percent_bully = bully / len(result) * 100
    percent_notbully = notbully / len(result) * 100
    labels = ['Bully', 'Not Bully']
    sizes = [percent_bully, percent_notbully]
    colors = ['#004d4d','#009973']
    explode = (0.2, 0)
    patches = plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'color': "w", 'fontsize': 14})
    plt.legend(labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    return plt,bully,notbully


def account_analysis(username):
    live_data = get_tweets(username)
    followers = live_data['followers'][0]
    following = live_data['following'][0]
    profile = live_data['profile_img'][0]
    clean_live_data = preprocess(live_data)
    clean_live_data_xtest = clean_live_data['text']
    clean_live_data_xtest.drop_duplicates(keep='first', inplace=True)
    result = model(clean_live_data_xtest)
    plt,bully,notbully = plot_pie(result)
    plt.savefig(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\pie.png', dpi=300, transparent=True)
    plt.close()
    values2 = [bully, notbully]
    labels2 = ['Bully', 'Not Bully']
    colors2 = ['#004d4d','#009973']
    plt.figure(figsize=(4, 4), dpi=70)
    plt.bar(labels2, values2, color=colors2, width=.7)
    plt.savefig(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\bar.png', dpi=300, transparent=True)
    plt.close()
    user = username
    print(user)
    print(followers)
    print(following)
    print(profile)
    return followers,user,following,profile


def perform_analysis(data1):
    data = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\FinalProject\PreprocessedDataset3.csv")
    data.dropna(inplace=True)
    x = data['text']
    y = data['label']
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 2))
    training_data_vect = tfidf_vect.fit_transform(x)
    clean_live_data_xtest = data1['text']
    test_data_vect = tfidf_vect.transform(clean_live_data_xtest)
    model = LogisticRegression()
    model.fit(training_data_vect, y)
    result = model.predict(test_data_vect)
    plt, bully, notbully = plot_pie(result)
    plt.savefig(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\trends_analysis.png', dpi=300, transparent=True)
    plt.close()
    total = bully+notbully
    return total,bully,notbully


def trending_hashtag():
    api = api_creation()
    trends1 = api.trends_place(2282863, lang='en')
    trends = list([trend['name'] for trend in trends1[0]['trends']])
    lang = []
    for x in trends:
        lng = detect(x)
        lang.append(lng)
    index = []
    for n, i in enumerate(lang):
        if i == 'en':
            index.append(n)
    eng_trends = []
    for x in index:
        eng_trends.append(trends[x])
    return eng_trends


def plot_wordcloud(eng_trends):
    word = ' '
    for words in eng_trends:
        word = word + words + ' '
    mask = np.array(Image.open(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\hashtagicon.png'))
    wordcloud = WordCloud(background_color='white', max_words=100, max_font_size=150, random_state=42, mask=mask)
    image_color = False
    wordcloud.generate(word)
    plt.figure(figsize=(6.0, 6.0))
    if image_color:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    else:
        plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\final_wordcloud.png', dpi=300, transparent=True)
    plt.close()


def get_text(eng_trends, loc):
    api = api_creation()
    search_term = eng_trends[loc]
    tweets_for_csv = []
    for tweet in tweepy.Cursor(api.search, q=search_term, lang='en').items(100):
        tweets_for_csv.append([search_term, tweet.text.encode("utf-8")])
    with open(r'C:\Users\LENOVO\PycharmProjects\FinalProject\tweets.csv', 'w+', encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(tweets_for_csv)
    with open(r'C:\Users\LENOVO\PycharmProjects\FinalProject\tweets.csv', errors='ignore') as fd:
        data = pd.read_csv(fd, error_bad_lines=False, names=["text"])
    return data


def getFriends(name):
    engine = create_engine('sqlite:///friendsdatabase.db', connect_args={'check_same_thread': False},poolclass=StaticPool)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session2 = DBSession()
    Fetch_Freinds_List(name)
    for i in session2.query(Friends.id).distinct():
        temp = session2.query(Friends).filter_by(mainuser=name).filter_by(id=i.id).first()
        if temp is not None:
            result = Result(mainuser=temp.mainuser, fid=temp.id, fname=temp.name, fscreen_name=temp.screen_name,
                            fprofile_image_url=temp.profile_image_url,
                            finalresult=temp.finalresult)
            session2.add(result)
            session2.commit()
    items = session2.query(Result).all()
    return items


def getFollowers(name):
    enginefollow = create_engine('sqlite:///followersdatabase.db', connect_args={'check_same_thread': False},
                                 poolclass=StaticPool)
    Base.metadata.bind = enginefollow
    DBSession = sessionmaker(bind=enginefollow)
    sessionfollow1 = DBSession()
    Fetch_Follower_List(name)
    for i in sessionfollow1.query(Followers1.id).distinct():

        temp = sessionfollow1.query(Followers1).filter_by(mainuser=name).filter_by(id=i.id).first()
        if temp is not None:
            result = Result(mainuser=temp.mainuser, fid=temp.id, fname=temp.name, fscreen_name=temp.screen_name,
                            fprofile_image_url=temp.profile_image_url,
                            finalresult=temp.finalresult)
            sessionfollow1.add(result)
            sessionfollow1.commit()
    items = sessionfollow1.query(Result).all()
    return items


PEOPLE_FOLDER = os.path.join(r'static', 'output_images')
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.secret_key = 'mysecret'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'sakshi'
app.config['MYSQL_DB'] = 'pythonlogin'

mysql = MySQL(app)


@app.route('/hh.html')
def hh():
    return render_template("hh.html")


@app.route('/DashboardMain.html')
def dash():
    return render_template("DashboardMain.html")


@app.route('/friends.html')
def fri():
    return render_template("friends.html")


@app.route('/friends.html', methods=['POST'])
def friendsanalysis():
    try:
        name = request.form['inputhandle']
        if name.isdigit():
            flash(
                "Please enter valid tweet containing character/string/text data as only numeric values as userhandle is invalid")
            return render_template("friends.html")
        elif name.isspace():
            flash(
                "Please enter valid tweet containing character/string/text data as only spaces as userhandle is invalid")
            return render_template("friends.html")
        elif re.match(r'^[_\W]+$', name):
            flash(
                "Please enter valid tweet containing character/string/text data as only special characters as userhandle is invalid")
            return render_template("friends.html")
        else:
            items = getFriends(name)
            return render_template('friendstable.html', title=name, items=items)
    except TweepError as e:
        if getExceptionMessage(e.reason) == "Sorry, that page does not exist.":
            flash("Please enter a valid twitter handle for analysis")
        else:
            flash("Please make sure you are connected to an active internet connection")
        return render_template("friends.html")
    except ProgrammingError:
        flash("Database Connection failed! Please try again")
        return render_template("followers.html")


@app.route('/followers.html')
def foll():
    return render_template("followers.html")


@app.route('/followers.html', methods=['POST'])
def getValue():
    try:
        name = request.form['inputhandle']
        if name.isdigit():
            flash(
                "Please enter valid tweet containing character/string/text data as only numeric values as userhandle is invalid")
            return render_template("followers.html")
        elif name.isspace():
            flash(
                "Please enter valid tweet containing character/string/text data as only spaces as userhandle is invalid")
            return render_template("followers.html")
        elif re.match(r'^[_\W]+$', name):
            flash(
                "Please enter valid tweet containing character/string/text data as only special characters as userhandle is invalid")
            return render_template("followers.html")
        else:
            items = getFollowers(name)
            return render_template('followerstable.html', title=name, items=items)
    except TweepError as e:
        if getExceptionMessage(e.reason) == "Sorry, that page does not exist.":
            flash("Please enter a valid twitter handle for analysis")
        else:
            flash("Please make sure you are connected to an active internet connection")
        return render_template("followers.html")
    except ProgrammingError:
        flash("Database Connection Failed! Please try again")
        return render_template("followers.html")


@app.route('/index_tweet_analysis.html')
def indexpage():
    result = "BULLY Spotted.. !!!"
    return render_template("index_tweet_analysis.html", name=result)


@app.route('/index_tweet_analysis.html', methods=['POST'])
def tweet_analyzer():
    live_tweet = request.form['inputtweet']
    if live_tweet.isdigit():
        flash("Please enter valid tweet containing character/string/text data as only numeric values as tweet is invalid")
        return render_template("index_tweet_analysis.html")
    elif live_tweet.isspace():
        flash("Please enter valid tweet containing character/string/text data as only spaces as tweet is invalid")
        return render_template("index_tweet_analysis.html")
    elif re.match(r'^[_\W]+$', live_tweet):
        flash("Please enter valid tweet containing character/string/text data as only special characters as tweet is invalid")
        return render_template("index_tweet_analysis.html")
    else:
        tweet_input = [live_tweet]
        df = pd.DataFrame({'text': tweet_input})
        df.to_csv(r'C:\Users\LENOVO\PycharmProjects\FinalProject\sample.csv')
        tweet1 = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\FinalProject\sample.csv')
        tweet = preprocess(tweet1)
        print(tweet)
        result = model(tweet_input)
        if result[0] == 1:
            print("Bully")
            label = "Demeaning"
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'bully.jpg')
        else:
            print("Non-Bully")
            label = "Innocuous"
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'notbully.jpg')
        label = "The tweet entered is used in "+label+" sense"
        result_message = "Analysis Result"
    return render_template("index_tweet_analysis.html", label=label, user_image=filename, result=result_message)


@app.route('/indexpage3.html')
def load_account_page():
    return render_template("indexpage3.html")


@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, public max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route('/result.html', methods=['POST'])
def account_analyzer():
    try:
        username = request.form['inputhandle']
        if username.isdigit():
            flash("Please enter valid tweet containing character/string/text data as only numeric values as userhandle is invalid")
            return render_template("indexpage3.html")
        elif username.isspace():
            flash("Please enter valid tweet containing character/string/text data as only spaces as userhandle is invalid")
            return render_template("indexpage3.html")
        elif re.match(r'^[_\W]+$', username):
            flash("Please enter valid tweet containing character/string/text data as only special characters as userhandle is invalid")
            return render_template("indexpage3.html")
        else:
            followers, user, following, profile = account_analysis(username)
        return render_template("result.html", followers=followers, username=user, following=following, profile_img=profile)
    except tweepy.TweepError as e:
        print(getExceptionMessage(e.reason))
        if getExceptionMessage(e.reason) == "status code = 404":
            flash("Please enter a valid twitter handle for analysis")
        else:
            flash("Please make sure you are connected to an active internet connection")
        return render_template("indexpage3.html")


@app.route('/trendanalysis.html', methods=["GET", "POST"])
def trend_analysis():
    try:
        if os.path.exists(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\final_wordcloud.png'):
            os.remove(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\final_wordcloud.png')
        eng_trends = []
        if request.method == "POST" and request.form["check1"] == "TRENDING HASHTAGS":
            eng_trends = trending_hashtag()
            plot_wordcloud(eng_trends)
        return render_template("trendanalysis.html", eng_hashtag=eng_trends, len1=len(eng_trends))
    except tweepy.TweepError:
        flash("Please make sure you are connected to an active net connection")
        return render_template("trendanalysis1.html")


@app.route('/analysis.html', methods=["GET", "POST"])
def trend_analyzer():
    try:
        if request.method == "POST" and request.form["check1"] == "CYBER BULLYING ANALYSIS":
            api = api_creation()
            trends1 = api.trends_place(2282863, lang='en')
            trends_name = list([trend['name'] for trend in trends1[0]['trends']])
            trends_count = list([trend['tweet_volume'] for trend in trends1[0]['trends']])
            lang = []
            for x in trends_name:
                lng = detect(x)
                lang.append(lng)
            index = []
            count = []
            for n, i in enumerate(lang):
                if i == 'en':
                    index.append(n)
            eng_trends = []
            for x in index:
                eng_trends.append(trends_name[x])
                count.append(trends_count[x])
            index1 = []
            key = []
            value = []
            for n, x in enumerate(count):
                if x is not None:
                    index1.append(n)
            for x in index1:
                key.append(eng_trends[x])
                value.append(count[x])
            loc = value.index(max(value))
            keyword = key[loc]
            loc1 = eng_trends.index(keyword)
            plt.switch_backend('agg')
            plt.xticks(fontsize=10, rotation=90)
            colors2 = ['#00264d', '#004d99', '#0073e6', '#3399ff', '#80bfff', '#b3d9ff']
            plt.bar(key, value, color=colors2)
            plt.savefig(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\trends_bar.png', dpi=300, transparent=True,bbox_inches='tight')
            plt.close()
            df = pd.DataFrame({'Hashtags': key, 'No. of Tweets': value})
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            tabla = table(ax, df, loc='center', colWidths=[.17] * len(df.columns))  # where df is your data frame
            tabla.auto_set_font_size(False)  # Activate set fontsize manually
            tabla.set_fontsize(18)  # if ++fontsize is necessary ++colWidths
            tabla.scale(3, 3)  # change size table
            plt.savefig(r'C:\Users\LENOVO\PycharmProjects\FinalProject\static\table.png', transparent=True, bbox_inches='tight')
            plt.close()
            data = get_text(eng_trends,loc1)
            preprocessed_data = preprocess(data)
            preprocessed_data.drop_duplicates(subset='text', keep='first', inplace=True)
            total, bully, notbully = perform_analysis(preprocessed_data)
        return render_template("analysis.html", bully = bully, notbully = notbully, hash = eng_trends[loc1])
    except tweepy.TweepError:
        flash("Please make sure you are connected to an active net connection")
        return render_template("trendanalysis1.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    error=None
    if request.method == "POST"and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        details = request.form
        username = details['username']
        password = details['password']
        email = details['email']
        cur = mysql.connection.cursor()
        qs="""SELECT * FROM accounts WHERE username = %s"""
        cur.execute(qs,(username,))
        account = cur.fetchone()
        # If account exists show error and validation checks
        if account:
            error='Account with this username already exists!'
        elif not re.match(r'^[\w.+\-]+[@](gmail.com|hotmail.com|yahoo.com|rediffmail.com|outlook.com|gov.in)', email):
           error='Invalid email address!'
        elif not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', password):
           error = 'Invalid Password! Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters'
        elif not re.match(r'(^[A-Z]+[a-z0-9_]{3,15}$)', username):
           error= 'Username must start with character or number!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cur.execute('INSERT INTO accounts (username, password, email) VALUES ( %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            error = 'You have successfully registered!'
        #return redirect(url_for('login'))
    return render_template('reg.html',error=error)


@app.route('/editP.html')
def editP():
    if 'loggedin' in session:
        return render_template("editP.html", username=session['username'])


@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
     session.pop('loggedin', None)
     session.pop('username', None)
     return redirect(url_for('login'))


@app.route('/', methods=['GET', 'POST'])
def login():
   error=None
   if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if (account[0]==username):
           session['loggedin']=True
           session['username'] = request.form['username']
           error='Logged in successfully!'
           return render_template('DashboardMain.html')
        else:
            error= 'Incorrect username/password!'
    # Show the login form with message (if any)
   return render_template('index.html', error=error)


@app.errorhandler(404)
def page_not_found(e):
    return 'something went wrong'

@app.errorhandler(500)
def page_not_found(e):
    return 'something went wrong'


@app.route('/forg', methods=['GET', 'POST'])
def forg():
    error = None
    if request.method == 'POST' and 'uname' in request.form and 'passs' in request.form and 'confrm' in request.form:
        # Create variables for easy access
        uname = request.form['uname']
        passs = request.form['passs']
        confrm = request.form['confrm']
        cursor = mysql.connection.cursor()
        if(passs==confrm):
          upquer="""UPDATE accounts SET password = %s where username = %s"""
          inputData=(confrm,uname)
          count=cursor.execute(upquer,inputData)
          mysql.connection.commit()
        else:
           return 'Check your both password didnot match. Try Again!'
        # If account exists show error and validation checks
        if count:
            error = 'password changed successfully!'
        else:
            error= 'No such account exists!'
    return render_template('forgot.html',error=error)


@app.route('/UserProfile.html', methods=['GET', 'POST'])
def userup():
    error = None
    rrr=None
    if request.method == 'POST' and 'uname' in request.form and 'passs' in request.form and 'confrm' in request.form:
        # Create variables for easy access
        uname = request.form['uname']
        passs = request.form['passs']
        confrm = request.form['confrm']
        cursor = mysql.connection.cursor()
        if (uname==session['username'] and confrm==passs):
          upquer="""UPDATE accounts SET password = %s where username = %s"""
          inputData=(confrm,uname)
          count=cursor.execute(upquer,inputData)
          mysql.connection.commit()
        else:
            return 'Something went Wrong! Check your user name and password!'
        # If account exists show error and validation checks
        if count:
            error = 'password updated successfully!'
        else:
            error= 'No such account exists!'
    return render_template('UserProfile.html',error=error,usern=session['username'])


if __name__ == "__main__":
    app.run(debug=True)