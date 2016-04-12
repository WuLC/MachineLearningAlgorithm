# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2016-04-12 15:53:02
# @Last modified by:   WuLC
# @Last Modified time: 2016-04-12 19:42:16
# @Email: liangchaowu5@gmail.com
# @Function: implementation of User-based collaborative filetering 
# @Referer: chaper 2 of the book 《programming-collective-intelligence》


# sample data for test
critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
		'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
		'The Night Listener': 3.0},
		'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
		 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
		 'You, Me and Dupree': 3.5},
		'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
		 'Superman Returns': 3.5, 'The Night Listener': 4.0},
		'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
		 'The Night Listener': 4.5, 'Superman Returns': 4.0,
		 'You, Me and Dupree': 2.5},
		'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
		 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
		 'You, Me and Dupree': 2.0},
		'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
		 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
		'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0,'Superman Returns': 4.0}}

import math

# four different methonds to caculate users' similarity
def user_similarity_on_euclidean(scores,user1,user2):
    """caculate similarity of two users based on Euclidean Distance
    
    Args:
        scores (dict{dict{}}): group of users' scores on some movies
        user1 (str): one of the user
        user2 (str): the other user
    
    Returns:
        float: reciprocal of Euclidean Distance(range(0,1)) between user1 and user2, the bigger the more similar
    """
    commom = [movie for movie in scores[user1] if movie in scores[user2]]
    if len(commom) == 0:  #no common item of the two users
        return 0
    total = sum([math.pow(scores[user1][movie] - scores[user2][movie], 2)
                 for movie in commom])
    similarity=math.sqrt(total)
    return 1/(total+1)


def user_similarity_on_cosine(scores,user1,user2):
    """caculate similarity of two users based on cosine similarity
    
    Args:
        scores (dict{dict{}}): group of users' scores on some movies
        user1 (str): one of the user
        user2 (str): the other user
    
    Returns:
        float: cosine similarity(range(-1,1)) between user1 and user2, the bigger the more similar
    """
    commom = [movie for movie in scores[user1] if movie in scores[user2]]
    if len(commom) == 0:  #no common item of the two users
        return 0

    pow_sum_1=sum([math.pow(scores[user1][movie], 2) for movie in commom])
    pow_sum_2=sum([math.pow(scores[user2][movie], 2) for movie in commom])
    multiply_sum=sum([scores[user1][movie] * scores[user2][movie] for movie in commom])
    if pow_sum_1 == 0 or pow_sum_2 == 0:
        return 0
    else:
        similarity = multiply_sum/math.sqrt(pow_sum_2*pow_sum_1)
        return similarity


def user_similarity_on_modified_cosine(scores, user1, user2):
    """caculate similarity of two users based on modified cosine similarity
    
    Args:
        scores (dict{dict{}}): group of users' scores on some movies
        user1 (str): one of the user
        user2 (str): the other user
    
    Returns:
        float: modified cosine similarity(range(-1,1)) between user1 and user2, the bigger the more similar
    """
    commom = [movie for movie in scores[user1] if movie in scores[user2]]
    if len(commom) == 0:  #no common item of the two users
        return 0
    average1 = float(sum(scores[user1][movie] for movie in scores[user1]))/len(scores[user1])
    average2 = float(sum(scores[user2][movie] for movie in scores[user2]))/len(scores[user2])
    # denominator
    multiply_sum = sum( (scores[user1][movie]-average1) * (scores[user2][movie]-average2) for movie in commom )
    # member
    pow_sum_1 = sum( math.pow(scores[user1][movie]-average1, 2) for movie in scores[user1] )
    pow_sum_2 = sum( math.pow(scores[user2][movie]-average2, 2) for movie in scores[user2] )
    
    modified_cosine_similarity = float(multiply_sum)/math.sqrt(pow_sum_1*pow_sum_2)
    return modified_cosine_similarity 


def user_similarity_on_pearson(scores, user1, user2):
    """caculate similarity of two users based on Pearson Correlation Coefficient
    
    Args:
        scores (dict{dict{}}): group of users' scores on some movies
        user1 (str): one of the user
        user2 (str): the other user
    
    Returns:
        float: Pearson Correlation Coefficient(range(-1,1)) between user1 and user2, the bigger the more similar
    """
    commom = [movie for movie in scores[user1] if movie in scores[user2]]
    if len(commom) == 0:  #no common item of the two users
        return 0
    average1 = float(sum(scores[user1][movie] for movie in scores[user1]))/len(scores[user1])
    average2 = float(sum(scores[user2][movie] for movie in scores[user2]))/len(scores[user2])
    # denominator
    multiply_sum = sum( (scores[user1][movie]-average1) * (scores[user2][movie]-average2) for movie in commom )
    # member
    pow_sum_1 = sum( math.pow(scores[user1][movie]-average1, 2) for movie in commom )
    pow_sum_2 = sum( math.pow(scores[user2][movie]-average2, 2) for movie in commom )
    
    modified_cosine_similarity = float(multiply_sum)/math.sqrt(pow_sum_1*pow_sum_2)
    return modified_cosine_similarity 


def find_similar_users(scores,user,similar_function = user_similarity_on_cosine):
    """find similar users based on the similar-function defined above
    
    Args:
        scores (dict{dict{}}): group of users' scores on some movies
        user (str): certain user 
        similar_function (function): certain similar-function defined above
    
    Returns:
        list[tuple]: list of users similar to the given user with their score
    """
    similar_users = [(similar_function(critics, user, otherUser), otherUser) for otherUser in scores if otherUser!=user]
    similar_users.sort()   # sort the users in terms of theri similarity score
    similar_users.reverse()
    # the above two lines are equal to : similar_users.sort(reverse = True)
    return similar_users 


def recommend_item(scores,user):
    """recommend items to user in terms of scores ,
       scores are caculated from similar users to the given user
    
    Args:
        scores (dict{dict{}}): group of users' scores on some movies
        user (str): certain user 
    
    Returns:
        list[tuple]: recommend items sorted in terms of their score
    """
    similar_users = find_similar_users(scores, user)
    swap_similar_users = {v:k for k, v in similar_users} # 交换键值，将存储kv对的列表转换为字典，交换后为无序
    all_movies = []
    for (k,v) in critics.items():
        for movie in v:
            if movie not in all_movies:
                all_movies.append(movie)
    item_score = []
    for movie in all_movies:
        score_sum = 0
        similarity_sum = 0
        for similarity, otherUser in similar_users:
            if critics[otherUser].has_key(movie):
                score_sum += critics[otherUser][movie] * similarity
                similarity_sum += swap_similar_users[otherUser]
        item_score.append((score_sum/similarity_sum, movie))

    item_score.sort(reverse=True)
    return item_score
 

if __name__ == '__main__':
    '''
    similarList = find_similar_users(critics, 'Lisa Rose')
    for i in similarList:
        print i
    '''
    item_score = recommend_item(critics,'Lisa Rose')
    for i,j in item_score:
        print i,j
    