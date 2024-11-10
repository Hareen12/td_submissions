import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from starter_code import LLM
# def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
# 	"""
# 	_______________________________________________________
# 	Parameters:
# 	words - 1D Array with 16 shuffled words
# 	strikes - Integer with number of strikes
# 	isOneAway - Boolean if your previous guess is one word away from the correct answer
# 	correctGroups - 2D Array with groups previously guessed correctly
# 	previousGuesses - 2D Array with previous guesses
# 	error - String with error message (0 if no error)

# 	Returns:
# 	guess - 1D Array with 4 words
# 	endTurn - Boolean if you want to end the puzzle
# 	_______________________________________________________
# 	"""

# 	# Your Code here
# 	# Good Luck!

# 	# Example code where guess is hard-coded
# 	guess = ["apples", "bananas", "oranges", "grapes"] # 1D Array with 4 elements containing guess
# 	endTurn = False # True if you want to end puzzle and skip to the next one

# 	return guess, endTurn
import spacy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModel
# from sklearn.cluster import KMeans
import torch
import numpy as np

# nlp = spacy.load("en_core_web_md")
# def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
#     """
#     Parameters:
#     words - 1D Array with 16 shuffled words
#     strikes - Integer with number of strikes
#     isOneAway - Boolean if your previous guess is one word away from the correct answer
#     correctGroups - 2D Array with groups previously guessed correctly
#     previousGuesses - 2D Array with previous guesses
#     error - String with error message (0 if no error)

#     Returns:
#     guess - 1D Array with 4 words
#     endTurn - Boolean if you want to end the puzzle
#     """
    
#     # Embed words using spaCy
#     print(words)
#     words = eval(words)
#     word_embeddings = [nlp(word).vector for word in words]
#     # print(word_embeddings)
#     # Use KMeans clustering to identify 4 groups (each group should ideally have 4 words)
#     num_clusters = 4
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(word_embeddings)
#     labels = kmeans.labels_

#     # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     # labels = kmeans.fit_predict(word_embeddings)
#     print(labels)
    
#     # Group words based on cluster labels
#     clusters = {}
#     for i, label in enumerate(labels):
#         if label not in clusters:
#             clusters[label] = []
#         clusters[label].append(words[i])
#     print(clusters)
#     # Convert clusters into a list of guesses
#     guesses = list(clusters.values())
    
#     # Determine which guess to return by checking previous guesses and one-away status
#     for guess in guesses:
#         if guess not in previousGuesses and len(guess) == 4:
#             # If previous guess was one word away, attempt slight modification
#             if isOneAway:
#                 guess = modify_guess_for_one_away(guess, words, previousGuesses)
#             return guess, False  # Continue guessing
    
#     # End turn if no new guesses are found or strikes are high
#     return [], True

# def modify_guess_for_one_away(guess, words, previousGuesses):
#     """
#     Modify guess slightly if we are one word away.
#     Parameters:
#     - guess: Current guess list
#     - words: All words in the puzzle
#     - previousGuesses: List of previous guesses
    
#     Returns:
#     - Modified guess list with one word replaced
#     """
#     # Try replacing one word from the guess with a word not yet guessed
#     for word in words:
#         if word not in guess:
#             modified_guess = guess[:-1] + [word]  # Replace the last word
#             if modified_guess not in previousGuesses:
#                 return modified_guess
#     return guess





# using transformers
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model1 = AutoModel.from_pretrained(model_name)
# def get_embedding(word):
#     inputs = tokenizer(word, return_tensors="pt")
#     outputs = model1(**inputs)
#     # Get the mean pooling of last hidden state (can be tuned)
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings.detach().numpy()

# def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
#     """
#     Parameters:
#     words - 1D Array with 16 shuffled words
#     strikes - Integer with number of strikes
#     isOneAway - Boolean if your previous guess is one word away from the correct answer
#     correctGroups - 2D Array with groups previously guessed correctly
#     previousGuesses - 2D Array with previous guesses
#     error - String with error message (0 if no error)

#     Returns:
#     guess - 1D Array with 4 words
#     endTurn - Boolean if you want to end the puzzle
#     """
#     print(words)
#     words = eval(words)
#     words = list(words)
#     # correctGroups = eval(correctGroups)
#     # correctGroups = list(correctGroups)
#     print(type(correctGroups))
#     print(correctGroups)
#     new_words =[]
#     for i in words:
#         f =0
#         for j in correctGroups:
#             if i in j:
#                 f =1 
#                 break
#         if f==0:
#             new_words.append(i)
#     words = new_words

    
#     word_embeddings = np.array([get_embedding(word).squeeze() for word in words])
#     num_clusters = len(words)//4
#     # num_clusters = 5

#     # hierarichial clustering
#     # agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
#     # labels = agg_clustering.fit_predict(word_embeddings)


#     # kmeans
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0,init="k-means++").fit(word_embeddings)
#     labels = kmeans.labels_
#     print(labels)
#     clusters = {i: [] for i in range(num_clusters)}
#     for i, label in enumerate(labels):
#         clusters[label].append(words[i])
#     print(clusters)

#     # balanced_clusters = []

#     # Create a function to balance clusters
#     # def balance_clusters(clusters):
#     #     # Flatten all words in the clusters into a single list of tuples (embedding, word)
#     #     all_words = []
#     #     for cluster_id, cluster_words in clusters.items():
#     #         for word in cluster_words:
#     #             embedding = get_embedding(word)
#     #             all_words.append((embedding, word))

#     #     # Apply KMeans again with the exact constraint of 4 words per group
#     #     # Re-run KMeans with a balanced approach
#     #     kmeans = KMeans(n_clusters=num_clusters, random_state=0, init="k-means++")
#     #     word_embeddings = [x[0] for x in all_words]
#     #     # a,b,c= word_embeddings.shape
#     #     # print(a,b,c)
#     #     # word_embeddings.reshape(a,c)
#     #     # print(word_embeddings.shape)
        
#     #     new_labels = kmeans.fit_predict(word_embeddings)
        
#     #     # Create the balanced clusters based on the new labels
#     #     balanced_clusters = {i: [] for i in range(num_clusters)}
#     #     for idx, label in enumerate(new_labels):
#     #         balanced_clusters[label].append(all_words[idx][1])

#     #     return balanced_clusters

#     # # Balance the clusters
#     # balanced_clusters = balance_clusters(clusters)
#     # guesses = balance_clusters
    


    
#     # Group words based on cluster labels
#     # clusters = {}
#     # for i, label in enumerate(labels):
#     #     if label not in clusters:
#     #         clusters[label] = []
#     #     clusters[label].append(words[i])
#     # print(clusters)
#     # # Convert clusters into a list of guesses
#     guesses = list(clusters.values())




    
#     # Determine which guess to return by checking previous guesses and one-away status
#     for guess in guesses:
#         if guess not in previousGuesses and len(guess) == 4:
#             # If previous guess was one word away, attempt slight modification
#             # if isOneAway:
#             #     guess = modify_guess_for_one_away(guess, words, previousGuesses)
#             return guess, False  # Continue guessing
#     #+1 cluster
#     kmeans = KMeans(n_clusters=num_clusters+1, random_state=0,init="k-means++").fit(word_embeddings)
#     labels = kmeans.labels_
#     print(labels)
#     clusters = {i: [] for i in range(num_clusters+1)}
#     for i, label in enumerate(labels):
#         clusters[label].append(words[i])
#     print(clusters)
#     guesses = list(clusters.values())
#     for guess in guesses:
#         if guess not in previousGuesses and len(guess) == 4:
#             # If previous guess was one word away, attempt slight modification
#             # if isOneAway:
#             #     guess = modify_guess_for_one_away(guess, words, previousGuesses)
#             return guess, False  # Continue guessing
    
#     #-1 cluster
#     if num_clusters-1==0:
#         return [], True
#     kmeans = KMeans(n_clusters=num_clusters-1, random_state=0,init="k-means++").fit(word_embeddings)
#     labels = kmeans.labels_
#     print(labels)
#     clusters = {i: [] for i in range(num_clusters-1)}
#     for i, label in enumerate(labels):
#         clusters[label].append(words[i])
#     print(clusters)
#     guesses = list(clusters.values())
#     for guess in guesses:
#         if guess not in previousGuesses and len(guess) == 4:
#             # If previous guess was one word away, attempt slight modification
#             # if isOneAway:
#             #     guess = modify_guess_for_one_away(guess, words, previousGuesses)
#             return guess, False  # Continue guessing

#     # End turn if no new guesses are found or strikes are high
#     return [], True





#using llms 
def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    Parameters:
    words - 1D Array with 16 shuffled words
    strikes - Integer with number of strikes
    isOneAway - Boolean if your previous guess is one word away from the correct answer
    correctGroups - 2D Array with groups previously guessed correctly
    previousGuesses - 2D Array with previous guesses
    error - String with error message (0 if no error)

    Returns:
    guess - 1D Array with 4 words
    endTurn - Boolean if you want to end the puzzle
    """
    # print(words)
    words = eval(words)
    words = list(words)
    # correctGroups = eval(correctGroups)
    # correctGroups = list(correctGroups)
    print(type(correctGroups))
    print(correctGroups)
    new_words =[]
    new_pq=[]
    for i in previousGuesses:
        new_pq.append(sorted(i))
    new_cq=[]
    for i in correctGroups:
        new_cq.append(sorted(i))
    correctGroups = new_cq
    for i in words:
        f =0
        for j in correctGroups:
            if i in j:
                f =1 
                break
        if f==0:
            new_words.append(i)
    words = sorted(new_words)
    l = LLM.LLM_model(words)

    for i in l:
        if i not in previousGuesses and len(i)==4:
            if i not in correctGroups:
                return i,False

    return [],True
    







# words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew', 'kiwi', 'lemon', 'mango', 'nectarine', 'orange', 'papaya', 'quince', 'raspberry']
# words = "['apple', 'banana', 'mango', 'kiwi','elephant', 'tiger', 'panda', 'zebra','red', 'blue', 'green', 'yellow','soccer', 'basketball', 'baseball', 'tennis']"
# words = "['RADICAL', 'LICK', 'EXPONENT', 'SHRED', 'GNARLY', 'ROOT', 'OUNCE','TWISTED', 'THRONE', 'TRACE', 'BATH', 'BENT', 'REST', 'POWDER', 'POWER','WARPED']"
# isOneAway = False
# correctGroups = "[]"
# previousGuesses = []
# error = "0"
# strikes =0

# guess, endTurn = model(words, strikes, isOneAway, correctGroups, previousGuesses, error)
# print("Guess:", guess)
# print("End Turn:", endTurn)


