
# Mini-Project_ Bollywood Movie Recommendation


####  Introduction: 



A Movie Recommendation system is a system that provides movie suggestions to users based on some dataset. Such a system will predict what movies a user will like based on the attributes of previously liked movies by that user. Content-Based recommendations have long been in fashion but they tend to overlook some great suggestions that may not be covered by mere  similarities. To overcome such shortcomings, we will combine collaborative  techniques having a  with neural networks to provide users(who have already rated movies previously) with appropriate suggestions.


#####  Approach:

The most common approach towards recommendation systems was the Content-based recommendations. Content-based techniques develop representations of clients and items through the investigation of additional data, for example, record content, client proles and the traits of items, to make suggestions


#### Problem Faced:  
Difficulty in  Exploring resources for Bollywood Movies Data Set .

Objective of the project:
To provide a system supporting features for searching, Suggesting  resources for Bollywood Movies.

####  Content-Based Filtering: 

Content-based filtering, also referred to as cognitive filtering, recommends items based on a comparison between
the content of the items and a user profile. The content of each item is represented as a set of descriptors or terms,
typically the words that occur in a document. The user profile is represented with the same terms and built up by
analyzing the content of items which have been seen by the user. 

 #### Memory-Based Collaborative Filtering:
 
Memory based algorithms approach the collaborative filtering problem
by using the entire database. Here we draw the similarity
between User-User or Item-Item by finding out the distance
between them. Distance is calculated by referring to some
numeric value. For use case of movie recommendation,
rating can be considered as a factor to calculate the distance.

 #### Limitation of Memory based filtering
 
Memory-based collaborative filtering approaches that
compute distance relationships between items or users have
these two major issues:

•  It does not scale particularly well to massive datasets,
   especially for real-time recommendations based on user
   behavior similarities which takes a lot of computations.
•  Ratings matrices may be overfitting to noisy representations of user tastes and preferences. When we use
   distance based neighborhood approaches on raw data,
   we match to sparse low-level details that we assume
   represent the users preference vector instead of the
   vector itself.

#### Requirements:
1.	Modules for :   Searching, Issuing, Returning and Buying resources 
2.	Provide intuitive, user friendly, software which could be access using any handheld device.
3.	Provide assistance to use this software and also prevent any inconsistency in the operations.


#### System development

1:  we will consider the Movies small dataset, and we focus on two files.
i.e., the movies.csv and ratings.csv.

Movies.csv has three fields namely: 
1.	MovieId – It has a unique id for every movie
2.	Title – It is the name of the movie
3.	Genre – The genre of the movie

The ratings.csv file has four fields namely:
1.	Userid – The unique id for every user who has rated one or multiple movies
2.	MovieId – The unique id for each movie
3.	Rating – The rating given to a user to a movie
 
2: We Change the working directory and replace it with  our dataset is stored in it .

 3: We Read the ratings file with the below command into the local variable ratings  shows you the top five records in the data set. We see that we are using the pandas library in the cell 
 
4: Next, we plot a bar graph describing the total number of reviews for each movie individually.
5: Finally, we arrange the titles along with their ratings in decending order. This gives us a list of top-rated movies

 

##### 	Software development tools
1.	Microsoft Edge
2.	HTML/CSS/JAVA SCRIPT
3.	Draw.io
4.	CHROME 
5.	API
6.	Jupiter Notebook
            Pandas
            A Python scikit
            
##### REFERENCES
[1] https://www.datacamp.com/community/tutorials/recommender-systems python

[2] https://towardsdatascience.com/various-implementations-of collaborative-filtering-100385c6dfe0

[3] https://medium.com/@james aka yale/the-4-recommendation-engines that-can-predict-your-movie-tastes-bbec857b8223

[4] https://en.wikipedia.org/wiki/Singular value decomposition

[5] https://github.com/gpfvic/IRR

[6] https://surprise.readthedocs.io/en/stable/index.html

[7] https://www.quora.com/Whats-the-difference-between-SVD-andSVD++

[8] https://blog.statsbot.co/singular-value-decomposition-tutorial 52c695315254

[9] https://www.quora.com/Whats-the-difference-between-SVD-and SVD++

[10] https://medium.com/recombee-blog/machine-learning-for recommender-systems-part-1-algorithms-evaluation-and-cold-start 6f696683d0ed

[11] http://www.awesomestats.in/python-recommending-movies/

[12] https://ieeexplore.ieee.org/document/8058367

[13] http://recommender-systems.org/content-based-filtering/

[14] https://hackernoon.com/the-fastest-way-to-identify-keywords-in-news articles-tfidf-with-wikipedia-python-version-baf874d7eb16

[15] https://www.researchgate.net/publication/21547071

