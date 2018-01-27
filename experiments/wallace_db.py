# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:14:48 2016

@author: mattia

WallaceDBHelper
This class wraps all the operations needed to use the Wallace dataset
contained in the sqlite db.
It provides the client with methods for retrieving texts and labels, 
plus the other operations made by paper authors to compute their 
statistics.
"""
import sqlite3
import numpy as np

class WallaceDBHelper:
    
    def __init__(self):
        db_path = "../datasets/ironate.db"
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        labelers_of_interest = [2,4,5,6]
        self.labeler_id_str = self._make_sql_list_str(labelers_of_interest)
        
    def __del__(self):
        """Close connection to db"""
        self.conn.close()
        
        
    def _make_sql_list_str(self, ls):
        """Make a list of objects a list of sql strings"""
        return "(" + ",".join([str(x_i) for x_i in ls]) + ")"

    def grab_comments(self, comment_id_list, verbose=False):
        """Extract from db the comment texts of ids comment_id_list
        and return them as a list of texts"""        
        comments_list = []
        for comment_id in comment_id_list:
            self.cursor.execute("select text from irony_commentsegment where comment_id='%s' order by segment_index" % comment_id)
            segments = self._grab_single_element(self.cursor.fetchall())
            comment = " ".join(segments)
            if verbose:
                print(comment)
            comments_list.append(comment.encode('utf-8').strip())
        return comments_list

    def _get_entries(self, a_list, indices):
        return [a_list[i] for i in indices]
    
    def get_labeled_thrice_comments(self):
        """ get all ids for comments labeled >= 3 times """
        self.cursor.execute(
            '''select comment_id from irony_label group by comment_id having count(distinct labeler_id) >= 3;'''
        )
        thricely_labeled_comment_ids = self._grab_single_element(self.cursor.fetchall())
        return thricely_labeled_comment_ids

    def _grab_single_element(self, result_set, COL=0):
        """Return the values of a single column in a result_set"""
        return [x[COL] for x in result_set]
    
    def get_all_comment_ids(self):
        """Return all the comment ids"""
        comment_ids = self._grab_single_element(self.cursor.execute(
                    '''select distinct comment_id from irony_label where labeler_id in ?;''', 
                        self.labeler_id_str))
        return comment_ids
    
    def get_ironic_comment_ids(self):
        self.cursor.execute(
            '''select distinct comment_id from irony_label 
                where forced_decision=0 and label=1 and labeler_id in %s;''' % 
                self.labeler_id_str)
    
        ironic_comments = self._grab_single_element(self.cursor.fetchall())
        return ironic_comments
        
    def get_texts_and_labels(self, debug=False):
        """Extracts texts of labeled comments and their labels"""
        #Retrieve all labeled comments
        comments_id = self.get_labeled_thrice_comments()
        #Retrieve ironic comments
        ironic_comment_ids = self.get_ironic_comment_ids()
        #Retrieve the texts
        texts = self.grab_comments(comments_id)
        #Assign labels
        y = []
        for _id in comments_id:
            if _id in ironic_comment_ids:
                y.append(1)
            else:
                y.append(-1)
        if debug:
            print("Size of texts, labels: {}, {}".format(len(texts), len(y)))
        return np.array(texts), np.array(y)
        