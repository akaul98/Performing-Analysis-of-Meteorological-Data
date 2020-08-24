
Performing Analysis of Meteorological Data

In this project, I have found meteorological data for 10 cities of Maharashtra State. Meteorological parameters such as pressure, temperature, humidity, rain, etc. I  have cleaned the data set for analysis, performed data cleaning, performed analysis for tested for the Hypothesis which is the effect the proximity of the sea has on the climate, and finally put-forth my conclusion. 
this is the following steps I have performed:
  Choose a set of 10 cities that will serve as reference standards. These cities are selected in order to cover the entire range of the plain.  Having distances of 0 km to up-to 400 km from the sea. ( Easiest way is to approximate the distance of (South) Mumbai as 0 km, as we all know it's touching the Arabian sea. Hence you can find all cities up-to 400 km, from Mumbai.) 
  Now we have to determine the distances of these cities from the sea(I have used google maps to determine the distance between the cities ). The following 10 cities with their distances are as follows:
MUMBAI 0 km
PUNE 150 km 
NASHIK 170 km 
NAVI MUMBAI 25 km
VASAI - VIRAR 60 km 
PANVEL 35 km
AURANGABAD 340 km
BHIWANDI 34 km
KOLHPUR 380 km 
DHULE 320 km

Finding the Data Source 
Once the system under study has been defined, I have established a data source from which I have obtained the needed data. By browsing the Internet, I  discovered a site that provides meteorological data measured from various locations around the world.( https://www.meteoblue.com). And I have downloaded the CSV file of all the city-data. The data is taken on an hourly basis over a span of 8 days.  

jupyter notebook 
Now we can begin the cleaning and data visualization of the CSV file with the help of jupyther notebook.

The first step is to import all the important all headers and files which are required:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter
from sklearn.model_selection import train_test_split


Analysis of Processed Meteorological Data


 I have collected data from all the cities involved in the analysis and have processed and collected them in a data-frame and have saved the data in CSV file, then use the read_csv() function of pandas, I have converted CSV files to the data frame .

data = pd.read_csv("mumbai.csv") 
data1 = pd.read_csv("pune.csv") 
data2 = pd.read_csv("nashik.csv")
data4 = pd.read_csv("navi_mumbai.csv") 
data5 = pd.read_csv("pan.csv")
data6 = pd.read_csv("vasai.csv")
data7 = pd.read_csv("aurangabad.csv")
data8 = pd.read_csv("bhiwandi.csv") 
data9 = pd.read_csv("kolhapur.csv")
data10 = pd.read_csv("dhule.csv")





figure 1:The data-frame structure corresponding to a city

To make sure we have sufficient data to perform the analysis we use the shape() function to   check the number of entries in the data frame


print(data .shape) 
print(data1 .shape)
print(data2.shape)
print(data4.shape)
print(data5.shape)
print(data6 .shape)
print(data7 .shape)
print(data8 .shape)
print(data9.shape)
print(data10.shape)


This will give the following result:


(192, 8)
(192, 8)

(192, 8)

(192, 8)

(192, 8)
(192, 8)
(192, 8)
(192, 8)
(192, 8)
(192, 8)

All the properties that is related to the time of acquisition expressed from the reading of time and date is  converted with the help of type function column in the data-frame and needs to be converted into timestamp. 





data1['Hour'] = pd.to_datetime(data1['Hour'])   
data1['Year'] = pd.to_datetime(data1['Year'])   
time1=data1["Hour"].iloc[0]
time1.hour

data1["Hour"]=data1["Hour"].apply(lambda time1:time1.hour) 

Data Visualisation 

A normal way to approach the analysis of the data we have just collected is to use
data visualisation. The matplotlib library includes a set of tools to generate charts on
which to display data. In fact, data visualisation helps you a lot during data analysis to
discover some features of the system you are studying.

For example, a simple way to choose is to analyze the trend of the temperature during the day. For example, consider the city of Mumbai.

byMonth = data.groupby('Hour').mean()    
byMonth['Temperature  [2 m above gnd]'].plot() 

We have taken a mean of all-day temperature, total precipitation, wind speed, wind direction, and grouped them together on an hourly basis.

Graph  with temperature in y-axis and hour in the x-axis




As you can see, the temperature trend follows a nearly sinusoidal pattern characterized by a temperature that rises in the morning, to reach the maximum value during the heat of the afternoon (between 2:00 and 6:00 pm). Then, the temperature decreases to a minimum value corresponding to just before dawn, that is, at 6:00 am.

Then I have evaluated the trends of different cities simultaneously. This is the only way to see if the analysis is going in the right direction. Thus, choose the three cities closest to the sea and the three cities farthest from it.

plt.axis([0, 23, 0, 50])
byMonth10['Temperature  [2 m above gnd]'].plot(color="pink",label ="farthest city= dhule" )#dhule city
byMonth9['Temperature  [2 m above gnd]'].plot(color="pink",label ="farthest city= kolhapur " )#kolhapur
byMonth7['Temperature  [2 m above gnd]'].plot(color="pink",label ="farthest  city= aurangabad"  )#aurangabad
byMonth5['Temperature  [2 m above gnd]'].plot(color="black",label ="nearest city= panvel")#panvel
byMonth4['Temperature  [2 m above gnd]'].plot(color="black",label ="nearest city= navi-mumbai"  )#navi mumbai
byMonth['Temperature  [2 m above gnd]'].plot(color="black" ,label ="nearest city= mumbai" )#mumbai

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)#legend to make sure 





Figure 3: Temperature trend of six different cities (black is the closest to the sea; pink is the farthest)


Looking at Figure 3, the results seem promising. In fact, the three closest cities have maximum temperatures much lower than those farthest away, whereas there seems to be little difference in the minimum temperatures. In order to go deep into this aspect, I collected the maximum and minimum temperatures of all 10 cities and display a line chart that charts these temperatures compared to the distance from the sea.

distance= [data10["distance from sea[km]"].mean(),
    data9["Distance from sea [km]"].mean(),
    data8["distance from sea[km]"].mean(),
    data7["distance from sea [km]"].mean(),
    data6["distance from sea[km]" ].mean(),
     data5["distance from sea [km]"].mean(),
     data4["distance from sea [km]"].mean(),
     data2["distance from sea [km]"].mean(),
    data1["distance from sea[km]"].mean(),
    data["distance from sea [km]"].mean()]

temp_max = [data10['Temperature  [2 m above gnd]'].max(),
            data9['Temperature  [2 m above gnd]'].max(),
            data8['Temperature  [2 m above gnd]'].max(),
            data7['Temperature  [2 m above gnd]'].max(),
            data6['Temperature  [2 m above gnd]'].max(),
            data5['Temperature  [2 m above gnd]'].max(),
            data4['Temperature  [2 m above gnd]'].max(),
            data2['Temperature  [2 m above gnd]'].max(),
            data1['Temperature  [2 m above gnd]'].max(),

            data['Temperature  [2 m above gnd]'].max()]

temp_min = [data10['Temperature  [2 m above gnd]'].min(),
            data9['Temperature  [2 m above gnd]'].min(),
            data8['Temperature  [2 m above gnd]'].min(),
            data7['Temperature  [2 m above gnd]'].min(),
            data6['Temperature  [2 m above gnd]'].min(),
            data5['Temperature  [2 m above gnd]'].min(),
            data4['Temperature  [2 m above gnd]'].min(),
            data2['Temperature  [2 m above gnd]'].min(),
            data1['Temperature  [2 m above gnd]'].min(),

            data['Temperature  [2 m above gnd]'].min()]

So you can say that the average distance in which the effects of the sea vanish is 58 km. Now you can analyze the minimum temperatures.
plt.axis((0,400,15,25)) 
plt.plot(distance,temp_min,'bo') 

The result is shown in Figure 4.

In this case, it appears very clear that the sea has no effect on minimum temperatures recorded during the night, or rather, around six in the morning.

                              Figure 4: Trend of minimum temperature in relation to distance from the sea
  


plt.plot(dist,temp_max,'ro')

The result is shown in Figure 5.


                       Figure 5: Trend of maximum temperature in relation to distance from the sea


As shown in Figure 5 one can confirm that the hypothesis, that the presence of the sea somehow influences meteorological parameters is true. Furthermore, one can see that the effect of the sea decreases rapidly, and after about 60-70 km.
 An interesting thing would be to represent the two different trends with two straight lines obtained by linear regression.
To do this, you can use the SVR method provided by the scikit-learn library.

x = np.array(dis)#array of distance
y = np.array(temp_max)#array of maximum temperature 
x1 = x[x<100]
x1 = x1.reshape((x1.size,1))
y1 = y[x<100]
x2 = x[x>50]
x2 = x2.reshape((x2.size,1))
y2 = y[x>50]
from sklearn.svm import SVR
svr_lin1 = SVR(kernel='linear', C=1e3)
svr_lin2 = SVR(kernel='linear', C=1e3)
svr_lin1.fit(x1, y1)
svr_lin2.fit(x2, y2)
xp1 = np.arange(10,100,10).reshape((9,1))
xp2 = np.arange(50,400,50).reshape((7,1))
yp1 = svr_lin1.predict(xp1)
yp2 = svr_lin2.predict(xp2)
plt.plot(xp1, yp1, c='r', label='Strong sea effect')
plt.plot(xp2, yp2, c='b', label='Light sea effect')
plt.axis((0,400,27,45))
plt.scatter(x, y, c='k', label='data')


Figure 6. The two trends described by the maximum temperatures in relation to distance

Another meteorological measure contained in the 10 data frames is the humidity. 

The below codes shows the analysis of humidity over an hour.

byMonth1['Relative Humidity  [2 m above gnd]'].plot()

The result is shown in Figure 7.

                                             figure 7: Humidity trend of Mumbai during the day


Even for this measure,  the trend of the humidity during the day for the three cities closest to the sea and for the three farthest away.

byMonth10['Relative Humidity  [2 m above gnd]'].plot(color="pink",label ="farthest city= dhule" )
byMonth9['Relative Humidity  [2 m above gnd]'].plot(color="pink",label ="farthest city= kolhapur " )
byMonth7['Relative Humidity  [2 m above gnd]'].plot(color="pink",label ="farthest  city= aurangabad"  )
byMonth5['Relative Humidity  [2 m above gnd]'].plot(color="black",label ="nearest city= panvel")
byMonth4['Relative Humidity  [2 m above gnd]'].plot(color="black",label ="nearest city= navi-mumbai"  )
byMonth['Relative Humidity  [2 m above gnd]'].plot(color="black" ,label ="nearest city= mumbai" )
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)




figure 8:The trend of the humidity of six different cities (black is the closest to the sea; pink is the farthest)


At first glance, it would seem that the cities closest to the sea experience more humidity than those farthest away and that this difference in moisture (about 20%) extends throughout the day. Let’s see if this remains true when we report the maximum and minimum humidity with respect to the distances from the sea.

max_humidity= [data10['Relative Humidity  [2 m above gnd]'].max(),
            data9['Relative Humidity  [2 m above gnd]'].max(),
            data8['Relative Humidity  [2 m above gnd]'].max(),
            data7['Relative Humidity  [2 m above gnd]'].max(),
            data6['Relative Humidity  [2 m above gnd]'].max(),
            data5['Relative Humidity  [2 m above gnd]'].max(),
            data4['Relative Humidity  [2 m above gnd]'].max(),
            data2['Relative Humidity  [2 m above gnd]'].max(),
            data1['Relative Humidity  [2 m above gnd]'].max(),
            data['Relative Humidity  [2 m above gnd]'].max()]

plt.plot(distance,max_humidity,'bo')

The result is shown in Figure 9.


figure 9:The trend of the maximum humidity as a function of distance from the sea 


min_humidity=[data10['Relative Humidity  [2 m above gnd]'].min(),
            data9['Relative Humidity  [2 m above gnd]'].min(),
            data8['Relative Humidity  [2 m above gnd]'].min(),
            data7['Relative Humidity  [2 m above gnd]'].min(),
            data6['Relative Humidity  [2 m above gnd]'].min(),
            data5['Relative Humidity  [2 m above gnd]'].min(),
            data4['Relative Humidity  [2 m above gnd]'].min(),
            data2['Relative Humidity  [2 m above gnd]'].min(),
            data1['Relative Humidity  [2 m above gnd]'].min(),
            data['Relative Humidity  [2 m above gnd]'].min()]




  figure 10 The trend of the minimum humidity as a function of distance from the sea. 

Looking at Figures 9 and 10, you can certainly see that the humidity is higher, both the minimum and maximum, in the city closest to the sea. 

                                                     Conclusion

From the above calculation, facts and figures we can conclude that the given hypothesis is valid and the sea does affect the meteorological data of a place, and cities closest to the sea experience more humidity than those farthest away.



