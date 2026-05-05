import pandas as pd
import numpy as np
from datetime import date, timedelta

np.random.seed(42)
N = 10000

hotels         = np.random.choice(['City Hotel','Resort Hotel'], N, p=[0.65,0.35])
arrival_years  = np.random.choice([2016,2017,2018,2019], N, p=[0.10,0.30,0.40,0.20])
arrival_months = np.random.choice(
    ['January','February','March','April','May','June',
     'July','August','September','October','November','December'],
    N, p=[0.04,0.04,0.07,0.08,0.09,0.10,0.12,0.13,0.10,0.08,0.08,0.07])
arrival_days   = np.random.randint(1,29,N)

lead_time = np.where(
    hotels=='City Hotel',
    np.random.exponential(120,N).clip(0,500).astype(int),
    np.random.exponential(60,N).clip(0,300).astype(int))

month_adr = {'January':60,'February':65,'March':75,'April':80,'May':90,
             'June':105,'July':120,'August':135,'September':110,
             'October':85,'November':70,'December':80}
base_adr = np.array([month_adr[m] for m in arrival_months])
adr = (base_adr + np.random.normal(0,20,N)).clip(20,500).round(2)
adr = np.where(hotels=='Resort Hotel', adr*1.15, adr).round(2)

weekend_nights = np.random.choice([0,1,2,3,4,5],N,p=[0.20,0.25,0.25,0.15,0.10,0.05])
week_nights    = np.random.choice([0,1,2,3,4,5,6,7],N,p=[0.05,0.20,0.25,0.20,0.15,0.08,0.05,0.02])
adults   = np.random.choice([1,2,3,4],N,p=[0.20,0.55,0.15,0.10])
children = np.random.choice([0,1,2,3],N,p=[0.75,0.15,0.07,0.03]).astype(float)
babies   = np.random.choice([0,1,2],  N,p=[0.92,0.06,0.02])

market_segs  = np.random.choice(
    ['Online TA','Offline TA/TO','Direct','Corporate','Groups','Complementary','Aviation'],
    N, p=[0.45,0.20,0.14,0.10,0.06,0.03,0.02])
deposit_type = np.random.choice(['No Deposit','Non Refund','Refundable'],N,p=[0.875,0.10,0.025])
customer_type= np.random.choice(['Transient','Transient-Party','Contract','Group'],N,p=[0.75,0.15,0.06,0.04])
meal         = np.random.choice(['BB','HB','FB','SC','Undefined'],N,p=[0.77,0.12,0.01,0.08,0.02])

c_list  = ['PRT','GBR','FRA','ESP','DEU','ITA','IRL','BEL','BRA','NLD',
           'USA','CHN','RUS','POL','SWE','AUT','CHE','NOR','DNK','FIN','Unknown']
c_probs = [0.26,0.10,0.08,0.07,0.06,0.05,0.04,0.03,0.03,0.03,
           0.03,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.04]
country = np.random.choice(c_list, N, p=c_probs)

prev_cancel   = np.random.choice([0,1,2,3],N,p=[0.85,0.10,0.04,0.01])
book_changes  = np.random.choice([0,1,2,3,4],N,p=[0.65,0.20,0.09,0.04,0.02])
wait_list     = np.random.choice([0,1,2,3,5,7,10,15,30],N,p=[0.80,0.04,0.03,0.03,0.02,0.02,0.02,0.02,0.02])
special_req   = np.random.choice([0,1,2,3,4,5],N,p=[0.37,0.32,0.18,0.08,0.03,0.02])
parking       = np.random.choice([0,1,2,3],N,p=[0.75,0.22,0.02,0.01])
prev_bookings = np.random.choice([0,1,2,3,4,5],N,p=[0.75,0.10,0.07,0.04,0.02,0.02])

cp  = np.full(N,0.15,dtype=float)
cp += (lead_time>150).astype(float)*0.18
cp += (deposit_type=='No Deposit').astype(float)*0.12
cp += (market_segs=='Online TA').astype(float)*0.10
cp += (hotels=='City Hotel').astype(float)*0.08
cp += (prev_cancel>0).astype(float)*0.15
cp += (adr>150).astype(float)*0.05
cp += (special_req==0).astype(float)*0.04
cp += (deposit_type=='Non Refund').astype(float)*(-0.25)
cp += (customer_type=='Contract').astype(float)*(-0.10)
cp += (market_segs=='Corporate').astype(float)*(-0.08)
cp  = cp.clip(0.02,0.96)
is_canceled=(np.random.random(N)<cp).astype(int)

res_status=np.where(is_canceled==1,'Canceled',
           np.where(np.random.random(N)<0.05,'No-Show','Check-Out'))
base_date=date(2015,7,1)
res_dates=[(base_date+timedelta(days=int(np.random.randint(0,1500)))).strftime('%Y-%m-%d') for _ in range(N)]
agent  =np.random.choice([0,1,2,3,4,5,6,7,8,9,10],N,
         p=[0.25,0.12,0.10,0.10,0.09,0.09,0.08,0.07,0.04,0.03,0.03]).astype(float)
company=np.random.choice([0,1,2,3,4],N,p=[0.82,0.07,0.05,0.04,0.02]).astype(float)

df=pd.DataFrame({
    'hotel':hotels,'is_canceled':is_canceled,'lead_time':lead_time,
    'arrival_date_year':arrival_years,'arrival_date_month':arrival_months,
    'arrival_date_week_number':np.random.randint(1,53,N),
    'arrival_date_day_of_month':arrival_days,
    'stays_in_weekend_nights':weekend_nights,'stays_in_week_nights':week_nights,
    'adults':adults,'children':children,'babies':babies,'meal':meal,'country':country,
    'market_segment':market_segs,
    'distribution_channel':np.random.choice(['Direct','TA/TO','Corporate','GDS','Undefined'],N,p=[0.14,0.64,0.13,0.07,0.02]),
    'is_repeated_guest':(prev_bookings>0).astype(int),
    'previous_cancellations':prev_cancel,'previous_bookings_not_canceled':prev_bookings,
    'reserved_room_type':np.random.choice(['A','B','C','D','E','F','G'],N,p=[0.50,0.10,0.06,0.18,0.07,0.05,0.04]),
    'assigned_room_type':np.random.choice(['A','B','C','D','E','F','G'],N,p=[0.50,0.10,0.06,0.18,0.07,0.05,0.04]),
    'booking_changes':book_changes,'deposit_type':deposit_type,
    'agent':agent,'company':company,'days_in_waiting_list':wait_list,
    'customer_type':customer_type,'adr':adr,
    'required_car_parking_spaces':parking,'total_of_special_requests':special_req,
    'reservation_status':res_status,'reservation_status_date':res_dates,
})

df.loc[np.random.choice(N,4,replace=False),'children']=np.nan
df.loc[np.random.choice(N,40,replace=False),'country']=np.nan
df.loc[np.random.choice(N,1300,replace=False),'agent']=np.nan
df.loc[np.random.choice(N,8600,replace=False),'company']=np.nan

df.to_csv('/home/claude/Hotel_Project/data/hotel_bookings.csv',index=False)
print("Saved:",df.shape,"| Cancel:",round(df['is_canceled'].mean()*100,1),"%")
