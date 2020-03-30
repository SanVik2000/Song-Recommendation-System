# Song-Recommendation-System
Recommending songs to the user based on his/her previous usage, by using K-Means clustering<br>
An **unsupervised** learning model which learns multiple-users playlists and provides a recommendation based on
- Genre
- Artist
- Both
## Description
The dataset is a dump of the [Free Music Archive (FMA)](https://freemusicarchive.org/), an interactive
library of high-quality, legal audio downloads. For more information regarding the dataset used, please click [here](https://github.com/mdeff/fma).<br>
## Usage
### Data-Processing
After having downloaded the dataset from the above link, we now have four important datasets, namely Echonest.csv, Features.csv, Genres.csv, Tracks.csv. We do some feature engineering and combine all the relevant and necessary information into one single entity. For doing this, execute the following commands:<br>
- ```cd Data_Collection_Exploration```
- ```python3 Data_Exploration.py```<br>
After successfull exeucution, you would be able to find two datasets, namely Final.csv and Metadata.csv.<br>
Download the processed datasets [here](https://drive.google.com/drive/folders/1FkZdcfzJklZ2zSwUfDxbQcJP8bKEQ699?usp=sharing).<br>
### Predicting Recommendations
We split the dataset into train and test samples. The test samples selects random songs from the processed data and creates a playlist which acts as the users history. Training sample corresponds to multiple user playlists. By using K-Means clustering, we recommend songs by executong the following commands:<br>
- ```cd Recommendation```
- ```python3 Recommendation.py```<br>
