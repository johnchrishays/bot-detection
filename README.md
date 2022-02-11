# Bod Detection / User Classification project for Twitter

## Overview

This project aims to use past datasets and fresh data to develop algorithms that can detect bots and non-human accounts on twitter.

 Original datasets used are stored in the ```/data``` folder (which was too big to upload). 
 
 Data processing was done in ```/data-processing```, where features are extracted from original user data, and processed ```.csv``` files are stored. 
 
 ```/baseline.ipynb``` contains a baseline single Random Forest clasifier for the processed data. 

 Different experiments are stored in folders containing ```experiment``` in their names.

## Data Processing 

Currently, the only dataset being processed is cresci-2017 (see [original paper](https://arxiv.org/abs/1701.03017) and [downloading link](https://botometer.osome.iu.edu/bot-repository/datasets.html)). The dataset was presented in groups of ```.csv``` files containing the user information and tweet history at the time of data collection, with each pair corresponding to a specific group of users. 

In the data processing process, I read in the ```.csv``` files, deleted excessive and unaligned columns with ```Pandas```, and calculated the features for each account by iterating through its personal information and tweets. The code dealing with this process is in ```/data-processing/cresci-2017-processing.ipynb```. 

After processing, data from different user groups are arranged and stored in different ```.csv``` files. Each line in the ```.csv``` file correspond to an account.

1. ```<unnamed>```: the ID of the user in the original file
2. ```status```: the number of statuses the user has generated
3. ```followers```: the number of followers the user has
4. ```friend```: the number of friends the user followed
5. ```favorite```: the number of "favorites" (now "likes") the user gave
6. ```listed```: the number of lists the user is on
7. ```follower_over_friend```: calculated by ```follower / friend```, zero if no friend
8. ```follower_minus_firned```: calculated by ```follower - friend```
9. ```favorite_over_friend```: calculated by ```favorite / friend```, zero if no friend
10. ```status_over_friend```: calculated by ```status / friend```, zero if no friend
11. ```url```: one if the user has a url, zero otherwise
12. ```url_facebook```: one if "facebook" is a substring of the user url, zero otherwise
13. ```url_instagram```: one if "instagram" is a substring of the user url, zero otherwise
14. ```url_blog```: one if "blog" is a substring of the user url, zero otherwise
15. ```time_zone```: one if the user has set a time zone, zero otherwise
16. ```location```: one if the user has set a location, zero otherwise
17. ```geo```: one if the user has enabled geographic info (this value is deprecated in current version of Twitter API), zero otherwise
18. ```verified```: one if the user is verified, zero otherwise
19. ```protected```: one if the user is protected, zero otherwise
20. ```bio```: one if the user has a bio
21. ```bio_len```: the length of the bio
22. ```bio_hashtag```: the number of hashtags "#" in the bio
23. ```bio_link```: the number of links "t.co" in the bio
24. ```bio_bot```: the number of "bot" substrings in bio
25. ```bio_official```: the number of "official" substrings in bio
26. ```bio_account```: the number of "account" substrings in bio
27. ```bio_emoji```: the number of emojis in bio
28. ```name_number```: the number of digits in the user's name
29. ```name_upper```: the number of capital letters in the user's name
30. ```name_space```: the number of spaces in the user's name
31. ```handle_number```: the number of digits in the user's handle
32. ```name_handdle_diff```: the likeliness between name and handle (calculated by Levenshtein distance)
33. ```retweet_mean```: the mean value of retweets of the user's tweets
34. ```retweet_std```: the standard deviation of retweets of the user's tweets
35. ```reply_mean```: the mean value of replies of the user's tweets
36. ```reply_std```: the standard deviation of replies of the user's tweets
37. ```reply_median```: the median value of replies of the user's tweets
38. ```favorite_mean```: the mean value of favorites of the user's tweets
39. ```favorite_std```: the standard deviation of favorites of the user's tweets
40. ```favorite_median```: the median value of favorites of the user's tweets
41. ```url_mean```: the mean number of urls in the user's tweets
42. ```url_std```: the standard deviation of number of urls in the user's tweets
43. ```hashtag_mean```: the mean number of hashtags in the user's tweets
44. ```hashtag_std```: the standard deviation of number of hashtags in the user's tweets
45. ```mention_mean```: the mean number of mentions in the user's tweets
46. ```mention_std```: the standard deviation of number of mentions in the user's tweets
47. ```retweeted_percentage```: how much percentage of the user's tweets are retweets
48. ```sensitive```: how much percentage of the user's tweets have links marked "sensitive" by Twitter
49. ```place```: how much percentage of the user's tweets have place information
50. ```tweet_time_interval_mean```: the mean value of the time interval between two tweets by the user
51. ```tweet_time_interval_std```: the standard deviation of the time interval between two tweets by the user

## Data Collection

Currently, I am collecting institutional accounts through two methods:

1. Collect users from top search results of searching college names, saved in ```/data-processing/college-official-accounts/100-schools.csv``` (data here are collected by searching US News top 100 universities and liberal arts colleges);
2. Collect users from lists that college official accounts are on, saved in ```/data-processing/college-official-accounts/60-lists-schools-organizations.csv``` (data here are collected from manually-gathered lists in ```twitter-lists-of-schools.csv```);

Both are, however, challenged by data pollution by real human users.

## ESC (Ensembles of Specialized Classifiers) Experiment One

In this experiment, I utilized data from cresci-2017 that includes genuine accounts, fake followers, social spambots, and traditional spambots. I trained one RF (Random Forest) aimed to distinguish genuine accounts from bot ones first. Then, I trained one RF to distinguish between the genuine accounts and each kind of bot accounts. During classification, two RFs with a highest confidence that the input is a bot or a genuine human was recorded. The bot confidence was prioritized such that if its predicted bot probability exceeds 0.5, the input will be classified as a bot, and this confidence is used to calculated the accuracy. On the other hand, the genuine confidence was used to calculate the accuracy, while the input is classified as a human.