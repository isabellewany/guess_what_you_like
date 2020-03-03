import numpy as np
import matplotlib.pyplot as plt
import sqlite3 
from tqdm import tqdm
import random

def scorer(x, y, metrics='cosine'):
    """
        Get the similarity score between x and y.
        
        Parameters
        ----------
        x, y: numpy.ndarray/list/dict
            The items to be compared.
        metrics: str
            The metrics used for measuring the similarity.
            Values in {'pearson', 'euclidean', 'cosine}, default as 'pearson'.
            
        Returns
        -------
        score: float
            The similarity score between x and y calculated by given metrics.
        
        Note
        -----
        If x, y are in format dict, then we compare their items with the same keys.
        Otherwise, we use the convention that their items are in order, i.e. x[i] corresponds with y[i].
            
    """
    #get the items to be compared according to different types of x and y.
    assert  type(x) == type(y)
    if type(x) == dict:
        items = [item for item in x if item in y]
    else:
        items = [i for i in range(min(len(x), len(y)))]
    #if there are no common items between x and y, return 0.
    if len(items) == 0:
        score = 0
    else:
        #Calculate the Pearson Correlation score.
        if metrics == 'pearson':
            sx = sum([x[_] for _ in items])
            sx2 = sum([x[_]**2 for _ in items])
            sy = sum([y[_] for _ in items])
            sy2 = sum([y[_]**2 for _ in items])
            sxy = sum([x[_]*y[_] for _ in items])

            a = sxy - sx * sy / len(items)
            b = np.sqrt((sx2-(sx**2)/len(items)) * (sy2-(sy**2)/len(items)))
            if b == 0:
                score = 0
            else:
                score = min(a/b, 1.0)
                
        #Calculate the Euclidean Distance score.
        elif metrics == 'eucildean':
            s = sum([(x[_] - y[_])**2 for _ in items])
            score = 1/(1+np.sqrt(s))
            
        #Calculate the Cosine score.
        elif metrics == 'cosine':
            sxy = sum([x[_]*y[_] for _ in items])
            sx2 = sum([x[_]**2 for _ in items])
            sy2 = sum([y[_]**2 for _ in items])
            if sx2 == 0 or sy2 == 0:
                score = 0
            else:
                score = min(sxy/(np.sqrt(sx2)*np.sqrt(sy2)), 1.0)
        
        else:
            raise KeyError('metrics should be \'pearson\', \'euclidean\' or \'cosine\'')
        
    return score


def scatter_scores(scores, a, b, users=None):
    """
        Scatter the scores[a] and scores[b] of assigned users.
        
        Parameters
        ----------
        scores: dict
            The dict of scores with user ID as keys and scores as values.
        a, b: str
            The assigned movies names.
        users: list/numpy.array
            The users to be scattered, keys of scores.
            Default as all users.
    """
    if users is None:
        users = list(scores.keys())

    items_exist = []#To make sure the annotations not overlap.
    for user in users:
        score = scores[user]
        try:
            sa, sb = score[a], score[b]
            if (sa, sb) not in items_exist:
                items_exist.append((sa, sb))
                plt.scatter(sa, sb)
                plt.annotate(user, (sa, sb))
            else:
                for i in range(1, 4):
                    if (sa-i*.2, sb+i*.2) not in items_exist:
                        break
                items_exist.append((sa-i*.2, sb+i*.2))
                #try increase the y of annotation as y+(1, 2, 3)
                plt.annotate(user, (sa-i*.2, sb+i*.2))
            plt.xlabel(a)
            plt.ylabel(b)

        #In case that the current user doesn't evaluate this movie.
        except KeyError:
            continue
        
def get_sim_user(user_tar, scores, n=3, metrics='cosine'):
    """
        Get the ID of similar users with given user.

        Parameters
        ----------
        user_tar:  str
            The target user ID.
        scores:
            See scatter_scores().
        n: int
            The number of similar users to be selected.
            Default as 3.
            If n == -1, then return all the similar users.
        metrics:
            See scorer().

        Returns
        -------
        sim_users: list
            The top n similar users' ID with given user.

    """
    #get the similarity scores of target user with all the other users.
    user_others = [_ for _ in scores.keys() if _ != user_tar]
    sim_scores = [(scorer(scores[user_tar], scores[_], metrics=metrics), _)\
                           for _ in user_others]
    sim_scores.sort(reverse=True)
    sim_users = sim_scores if n == -1 else sim_scores[: n]

    return sim_users


def recommend(user_tar, scores, n=3, metrics='cosine', train_all=False):
    """
        Generate a recommendation list for target user.
        
        Parameters
        ----------
        see get_sim_user()
        train_all: bool
            If True, then train all the dataset.
            If False, just generate the scores which user_tar hasn't scored.
        
        Returns
        -------
        A ordered list with n recommended movies as keys and the correspoding weighted score as values.
        
    """
    rcm_scores = {}
    norm_scores = {}
    
    sim_scores = get_sim_user(user_tar, scores, n=-1, metrics=metrics)
    for (sim_score, user_other) in sim_scores:
        #Neglect the person with the opposite tendance of tastes(sim_score<0)
            #and the uncorrelated persons(sim_score=0).
        if sim_score <= 0:
            continue
        else:
            for movie, score_other in scores[user_other].items():
                if ~train_all:
                    if movie in scores[user_tar]:#If this person has scored for this movie.
                        continue
                rcm_scores.setdefault(movie, 0.0)
                rcm_scores[movie] += sim_score*score_other#weighted by sim_score.
                norm_scores.setdefault(movie, 0.0)
                norm_scores[movie] += sim_score#used for normalization.
    
    rcms = [(rcm_score/norm_scores[_], _) for _, rcm_score in rcm_scores.items()]
    rcms.sort(reverse=True)
    rcms = rcms if n == -1 else rcms[: n]
    
    return rcms


class back_prop(object):
    def __init__(self, dbname):
        self.conn = sqlite3.connect(dbname, timeout=15, isolation_level=None)
        self.users = []
        self.input = []
        self.output = []
        self.weights = []
        self.movies_rcms = []
    
    def __del__(self):
        self.conn.close()
        
    def create_tables(self):
        #create tables
        self.conn.execute('create table users(user)')
        self.conn.execute('create table movies(movie)')
        self.conn.execute('create table scores(user, movie, score)')#the scores that user has evaluated.
        self.conn.execute('create table sim_scores(user, user_other, sim_score)')#the similarity score between user and user_other.
        self.conn.execute('create table weights(user, movie, weight)')#the weight between user and movie, serving for back_prop.
        self.conn.commit()
       
    def add_items(self, scores):
        #add a new item/items to database.
        for user, d_scores in scores.items():
            cur = self.conn.execute('select * from users where user=?', (user, ))
            res = cur.fetchone()
            if res is None:
                self.conn.execute('insert into users(user) values(?)', (user, ))
               
            for movie, score in d_scores.items(): 
                res = self.conn.execute('select * from movies where movie=?', (movie, )).fetchone()
                if res is None:
                    self.conn.execute('insert into movies(movie) values(?)', (movie, ))
                
                res = self.conn.execute('select * from scores where user=? and movie=?', (\
                                            user, movie)).fetchone()
                if res is None:
                    self.conn.execute('insert into scores(user, movie, score) values (?, ?, ?)', (\
                                            user, movie, score))
        self.conn.commit()
        
    def get_score_movie(self, user, movie):
        #get the score that target user has evaluated for movie.
        #if not scored before, then return 0.
        res = self.conn.execute('select score from scores where user=? and movie=?', \
                                (user, movie)).fetchone()
        if res is not None:
            return res[0]
        else:
            return 0
   
    
    def get_rcms(self, user, n=3, show_scores=True):
        #Get the top n recommendations for target user.
                
        #Query all the indexed movies.
        movies_indexed = list(self.conn.execute("select movie from movies"))
        movies_indexed = [_[0] for _ in movies_indexed]

        #Check if user is indexed.  If not, we could save the following steps to query the sim_scores and recommend randomly.
        res_user = self.conn.execute('select * from users where user=?', (user, )).fetchone()
        if res_user is None:
            self.conn.execute('insert into users(user) values(?)', (user, ))
            self.conn.commit()
            movies = random.sample(movies_indexed, len(movies_indexed))
            rcms = [(1.0, movie) for movie in movies]
         
        else:
            #For the indexed user.
            #obtain the list of movies that user hasn't evaluated before.
            movies_scored = list(set(list(self.conn.execute("select movie from scores where user=?", (user, )))))
            movies_scored = [_[0] for _ in movies_scored]
            
            #For an existing user who never scored any movie before, we return the recommedations randomly.
            if len(movies_scored) == 0:
                movies = random.sample(movies_indexed, len(movies_indexed))
                rcms = [(1.0, movie) for movie in movies]

            else:
                #Get all the other users in the database.
                user_others = list(self.conn.execute("select user from users where user != ?", (user, )))
                user_others = [user_other[0] for user_other in user_others]
                self.user_others = user_others
                movies_not_scored = set([_ for _ in movies_indexed if _ not in movies_scored])        

                sim_scores = []
                scores_user = {}
                for i, user_other in tqdm(enumerate(user_others)):
                    #Query the scores for movies_not_scored.
                    scores_user_other = {} #A dict with movies as keys and the correponding scores as values.
                    for movie in movies_scored:
                        if i ==0:
                            scores_user[movie]  = self.get_score_movie(user, movie)#Query the user's scores just once.
                        scores_user_other[movie] = self.get_score_movie(user_other, movie)

                    #Query the sim_score between user and user_other.
                    #if result is None, then calculate the sim_score between user and user_other and insert it into database.
                    res = self.conn.execute('select sim_score from sim_scores where (user=? and user_other=?) or (user=? and user_other=?)',
                                            (user, user_other, user_other, user)).fetchone()
                    if res is not None:
                        sim_score = res[0]
                    else:
                        sim_score = scorer(scores_user_other, scores_user)
                        self.conn.execute('insert into sim_scores(user, user_other, sim_score) values(?, ?, ?)', (user, user_other, sim_score))
                        self.conn.commit()
                    sim_scores.append((sim_score, user_other))

                self.sim_scores = sim_scores
                self.scores_user = scores_user

                #get recommendations
                rcm_scores = {}
                norm_scores = {}
                for (sim_score, user_other) in tqdm(sim_scores):
                    #Neglect the person with the opposite tendance of tastes(sim_score<0)
                        #and the uncorrelated persons(sim_score=0).
                    if sim_score <= 0:
                        continue
                    else:
                        #Calculate the scores for the movies that user havn't evaluated before.
                        for movie in movies_not_scored:
                            score_other = self.get_score_movie(user_other, movie)#Get the score of user_other for current movie.
                            rcm_scores.setdefault(movie, 0.0)
                            rcm_scores[movie] += sim_score*score_other#weighted by sim_score.
                            norm_scores.setdefault(movie, 0.0)
                            norm_scores[movie] += sim_score#used for normalization.

                rcms = [(rcm_score/norm_scores[_], _) for _, rcm_score in rcm_scores.items()]
                rcms.sort(reverse=True)
        
        self.rcms = rcms
        rcms = rcms[:n]
        #Return the scores and movies.
        if show_scores:
            return rcms
        
        #Return only the recommended movies.
        rcms = [_[1] for _ in rcms]
        return rcms
            
    
    def setup(self, user, n=3):
        #Set up the input and output nodes and the weight between two layers.
        
        #Input layer.
        self.user = user
        
        #Output layer.
        #check whether we have saved the weights in database.
        #we've saved which means we've already updated the weights.
        #In this case, we use the updated weight.
        rcms = self.conn.execute('select weight, movie from weights where user=?', (user, )).fetchall()
        if rcms is None:
            rcms = self.get_rcms(user, n, show_scores=True)
        
        self.movies_rcms = [_[1] for _ in rcms]#output nodes.
        #Initialize the nodes of input, output layer.
        self.input = 1.0
        self.output = [1.0] * len(self.movies_rcms)
        
        #set the weights of input->output layer as the simulated scores .
        weights = [_[0] for _ in rcms]#get the weight vector.
        #normalize the weights
        sum_weights = sum(weights)
        self.weights = [_/sum_weights for _ in weights]

    def feed_forward(self, user, n=3):
        self.input = 1.0#set the node of target user as 1.0.
        #set the activation status of the output layer.
        for i, movie in enumerate(self.movies_rcms):
            score =  self.input * self.weights[i]
            self.output[i] = np.tanh(score)
        return self.output[:]
            
    
    def back_prop(self, y, learning_rate=.5):
        #Calculate the error of output layer.
        err_out = []
        for i, movie in enumerate(self.movies_rcms):
            err = y[i] - self.output[i]
            err_out.append(d_tanh(self.output[i]) * err)#adjust the weight by multiplying the derivative of activation function.
            
        #update the weight.
        for i in range(len(self.weights)):
            adjust = err_out[i] * self.input
            self.weights[i] += learning_rate*adjust
          
        
    def update_db(self):
        #update the database.
        user = self.user
        for i, movie in enumerate(self.movies_rcms):
            res = self.conn.execute('select * from weights where user=? and movie=?', (user, movie)).fetchone()
            if res is None:
                self.conn.execute('insert into weights(user, movie, weight) values(?, ?, ?)', (user, movie, self.weights[i]))
            else:
                self.conn.execute('update weights set weight=? where user=? and movie=?', (self.weights[i], user, movie))
            self.conn.commit()
            
        
    def train(self, user, movies, selected_movie, n=3, learning_rate=.5):
        #Train the neural network using the feedback of users.
        self.setup(user, n)
        self.feed_forward(user, n)
        y = [0.0] * len(movies)
        y[movies.index(selected_movie)] = 1.0
        self.back_prop(y, learning_rate)
        self.update_db()
    
    
    
def d_tanh(x):
    #Calculate the derivative of tanh().
    return 1.0 - x**2
    
    
    
    