def cdata1(cdata):
    cdata = cdata.iloc[:,:].values
    import pandas as pd
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