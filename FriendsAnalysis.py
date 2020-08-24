
from tweepy import OAuthHandler,API,Cursor,TweepError
import LR_prediction
from datetime import datetime, date, time, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from friendsdatabase import Result,Friends,Base
from sqlalchemy.pool import StaticPool


consumer_key='hcoh0UTyaBrbCDP1QjKriOjXc'
consumer_secret='Jtw3TB8CmyzUTlHkt4UHhvg3zcM1KMm4RBM9N2ebBxOedgnx8o'
access_token='1136284891031597056-YLsUhgdABB6i7h4TSLOWvJFs7RlXxs'
access_token_secret='QKXYm4F2j8l9jbxUz9jLPVnWVcJAxs7WQTeYPGOa7Wq5v'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth, wait_on_rate_limit=True)

#,connect_args={'check_same_thread':False},poolclass=StaticPool

engine = create_engine('sqlite:///friendsdatabase.db',connect_args={'check_same_thread':False},poolclass=StaticPool)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
sessionfrnds = DBSession()



def getFriendsIds(username):
    i = 0
    friends = []

    for page in Cursor(auth_api.friends_ids, screen_name=username, wait_on_rate_limit=True, count=10).pages(20):
        try:
            friends.extend(page)
            i = i + 1
            if i == 1:
                return friends
        except TweepError as e:
            print("Going to sleep:", e)
            time.sleep(60)
    return friends


def Fetch_Freinds_List(username):
    x = getFriendsIds(username)
    sessionfrnds.query(Friends).delete()
    sessionfrnds.commit()
    sessionfrnds.query(Result).delete()
    sessionfrnds.commit()
    list = []
    i = 1
    for p in x:
        try:
            item = auth_api.get_user(p)
            fname = item.name
            id = str(item.id)
            profile_image_url = item.profile_image_url
            screen_name = item.screen_name
            statuses_count = str(item.statuses_count)
         
            if statuses_count=='0':
                frnds = Result(mainuser=username, fid=id, fname=fname,
                                   fscreen_name=screen_name, fprofile_image_url=profile_image_url,
                                   finalresult=0)
                sessionfrnds.add(frnds)
                sessionfrnds.commit()
                continue
           
            statuses = auth_api.user_timeline(id=item.id, count=50)
            for status in statuses:

                tweet = status.text.encode("utf-8")

                lg_decision = LR_prediction.lg_prediction(status.text)
                print(lg_decision)
                if lg_decision == 0:
                    final_decision = "Non-Bullying"
                else:
                    final_decision = "Bullying"

                friend=Friends(mainuser=username,id=id,name=fname,screen_name=screen_name,profile_image_url=profile_image_url,tweet=tweet, lg_result=final_decision)
                sessionfrnds.add(friend)
                sessionfrnds.commit()
                pass
            totaltweets = sessionfrnds.query(Friends).filter_by(id=id).count()
            x = sessionfrnds.query(Friends).filter_by(screen_name=screen_name).first()
            if totaltweets!=0:
                bullying=sessionfrnds.query(Friends).filter_by(id=id,lg_result="Bullying").count()
                if bullying!=0:
                    probab=(bullying/totaltweets)*100
                   
                else:
                    probab=0
                    
                x.finalresult=probab
                sessionfrnds.commit()
        except TweepError:
            print("Failed to run the command on that user, Skipping...")


