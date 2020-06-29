# Python-Projects
This is a repository for all of my Python projects. The background behind each project can be seen below.

### Amazon Reviews
| Project Name | Concepts | Data Source| Description 
| :-------:|:--------:| :-----------:| :--------------------:|
| [Reviews & Big Data Analytics](https://github.com/jessicakaye/Python-Projects/tree/master/Amazon/Reviews%20%26%20Big%20Data%20Analytics) | Text Processing, Latent Direchlet Allocation, Sentiment Analysis, Classification | Amazon review data within the "Clothing, Shoes, & Jewelry" category, as obtained from [Ni et al (2018) at UCSD](https://nijianmo.github.io/amazon/index.html) | LDA was conducted on the reviews for the top 10 most reviewed products within the category. The objective was to determine the dominant topics and words associated with each topic to infer what mattered the most to consumers for a specific product. Following this observation, sentiment analysis was conducted via SVM to analyze the change (in this case, lack thereof) in sentiment for a product based solely on reviews that contained the dominant topics.

### Spotify
| Project Name | Concepts | Data Source| Description 
| :-------:|:--------:| :-----------:| :--------------------:|
| [Song Popularity Classification](https://github.com/jessicakaye/Python-Projects/tree/master/Spotify/Song%20Popularity%20Classification) | API Requests, Data Cleaning, Classification | Spotify data for the following genres: pop, rap, rock, latin, hip hop, trap | Around 2000 songs were randomly pulled for each genre through Spotify's search query. The objective was to choose a machine learning model that would provide the highest accuracy in determining whether or not a song was classified as 'popular'. The threshold for popularity was determined based on the median popularity value. Algorithms included were KNN, CART, Naive Bayes, Random Forest, Logistic Regression, and SVM.

