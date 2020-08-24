from tweepy import OAuthHandler,API,Cursor,TweepError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from newdatabase1 import Result,Followers1,Base
import LR_prediction
from sqlalchemy.pool import StaticPool

engine = create_engine('sqlite:///followersdatabase.db',connect_args={'check_same_thread':False},poolclass=StaticPool)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
sessionfollow = DBSession()


consumer_key='hcoh0UTyaBrbCDP1QjKriOjXc'
consumer_secret='Jtw3TB8CmyzUTlHkt4UHhvg3zcM1KMm4RBM9N2ebBxOedgnx8o'
access_token='1136284891031597056-YLsUhgdABB6i7h4TSLOWvJFs7RlXxs'
access_token_secret='QKXYm4F2j8l9jbxUz9jLPVnWVcJAxs7WQTeYPGOa7Wq5v'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth, wait_on_rate_limit=True)


def getFollowerIds(username):
    i = 0
    followers = []

    for page in Cursor(auth_api.followers_ids, screen_name=username, wait_on_rate_limit=True, count=10).pages(20):
        try:
            followers.extend(page)
            i = i + 1
            if i == 1:
                return followers
        except TweepError as e:
            print("Going to sleep:", e)
    return followers

def Fetch_Follower_List(username):
    x = getFollowerIds(username)
    
    sessionfollow.query(Followers1).delete()
    sessionfollow.commit()
    sessionfollow.query(Result).delete()
    sessionfollow.commit()

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
                follower=Result(mainuser=username,fid=id,fname=fname,
                                    fscreen_name=screen_name,fprofile_image_url=profile_image_url,finalresult=0)
                sessionfollow.add(follower)
                sessionfollow.commit()
            else:

                statuses = auth_api.user_timeline(id=item.id, count=50)
                for status in statuses:

                    tweet = status.text.encode("utf-8")
                    print(tweet)
                    lg_decision = LR_prediction.lg_prediction(status.text)
                    if lg_decision==0:
                        final_decision="Non-Bullying"
                    else:
                        final_decision = "Bullying"

                    follower=Followers1(mainuser=username,id=id,name=fname,screen_name=screen_name,profile_image_url=profile_image_url,tweet=tweet,lg_result=final_decision)
                    sessionfollow.add(follower)
                    sessionfollow.commit()
                    pass
                totaltweets = sessionfollow.query(Followers1).filter_by(id=id).count()
                x = sessionfollow.query(Followers1).filter_by(screen_name=screen_name).first()
                if totaltweets!=0:
                    bullying=sessionfollow.query(Followers1).filter_by(id=id,lg_result="Bullying").count()
                    if bullying!=0:
                        probab=(bullying/totaltweets)*100
                        
                    else:
                        probab=0
                        
                    x.finalresult=probab
                    sessionfollow.commit()
        except TweepError:
            print("Failed to run the command on that user, Skipping...")

