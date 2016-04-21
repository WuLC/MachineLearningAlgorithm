# -*- coding: utf-8 -*-
# @Author: LC
# @Date:   2016-04-08 15:26:49
# @Last modified by:   WuLC
# @Last Modified time: 2016-04-21 14:41:10
# @Email: liangchaowu5@gmail.com
# @Function:implementation of decision tree described in programming-collective-intelligence in chapter 7
# @Referer: chapter 7 in book 《programming-collective-intelligence》

sample_data =[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]


class DecisionTreeNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def divid_set(rows, col, value):
    """divide the rows into two set according to the value of column col

    Args:
        rows (list[list]): rows to be divided
        col (int): certain column
        value (int or float or str): Description
        
    Returns:
        two sets in the form of lists that had been divided
    """
    set1, set2 = [], []
    if len(rows) == 0:
        return set1, set2

    if isinstance(value, int) or isinstance(value, float):
        for row in rows:
            if row[col] >= value:
                set1.append(row)
            else:
                set2.append(row)
    else:
        for row in rows:
            if row[col] == value:
                set1.append(row)
            else:
                set2.append(row)
    return set1, set2


def unique_counts(rows):
    """return the unique results of rows with their count in terms of the last column of the 
    
    Args:
        rows (list[list]): rows to process
    
    Returns:
        a dictionary to store the number of different resut appear in the last column of rows
    """
    count = {}
    for row in rows:
        res = row[-1]
        if res not in count:
            count[res] = 1
        else:
            count[res] += 1
    return count


def gini_impurity(rows):
    """caculate the gini impurity of rows
    
    Args:
        rows (list[list]): rows to be caculated the gini impurity
    
    Returns:
        float: gini impurity
    """
    count = unique_counts(rows)
    row_len = len(rows)
    gini_sum = 0
    for kind, num in count.items():
        p = float(num)/row_len
        gini_sum += p*(1-p)
    return gini_sum
    

def entropy(rows):
    """get the entropy of rows 
    
    Args:
        rows (list[list]): rows to be caculated the about their entropy
    
    Returns:
        float: entropy of the rows 
    """
    from math import log
    count = unique_counts(rows)
    ent = 0
    row_len = len(rows)
    for kind, num in count.items():
        p = float(num)/row_len
        ent -= p*log(p, 2)
    return ent

def variance(rows):
    """get the variance of the rows when the type is number
    
    Args:
        rows(list[list]): rows to be caculated their variance
    
    Returns:
        variance of the row in terms of the last row 
    """
    s = sum(row[-1] for row in rows)
    mean = s/len(rows)
    pow_sum = sum(pow(row[-1]-mean,2) for row in rows)
    variance = pow_sum/len(rows)
    reuturn variance
    
def build_tree(rows):
    """build the decision of the rows in the metric of entropy
    
    Args:
        rows (list[list]): build the tree with rows
    Returns:
        DecisionTreeNode: the root of the decision tree
    """
    col_num = len(rows[0])
    row_num = len(rows)
    original_entropy = entropy(rows)
    best_gain = 0
    best_col = None
    best_value = None
    tb_set = None
    fb_set = None
    for col in xrange(col_num-1):  # remove the last column
        col_features = []
        for row in rows:
            if row[col] not in col_features:
                col_features.append(row[col])
        for feature in col_features:
            set1, set2 = divid_set(rows, col, feature)
            p1 = float(len(set1))/row_num
            gain = original_entropy - p1*entropy(set1) - (1-p1)*entropy(set2)
            if gain > best_gain:
                best_gain = gain
                best_col = col
                best_value = feature
                tb_set = set1
                fb_set = set2
    if best_gain > 0:
        node = DecisionTreeNode(col=best_col, value=best_value)
        node.tb = build_tree(tb_set)
        node.fb = build_tree(fb_set)
    else:
        node = DecisionTreeNode(results=unique_counts(rows))
    return node


def print_tree(root, indent):
    """"print the decision tree
    
    Args:
        root (DecisionTreeNode): the root of the decision tree
        indent (str): space between root and branch
    
    Returns:
        None
    """
    if root.results is not None:
        print str(root.results)
    else:
        print root.col, ':', root.value, '?'
        print indent+'T->',
        print_tree(root.tb, indent+' ')
        print indent+'F->',
        print_tree(root.fb, indent+' ')


def classify(tree, observation):
    """classfy new instance with the already-built tree
    
    Args:
        tree (DecisoinTreeNode): decision that had been built
        observation (list): instance to be classfy
    
    Returns:
        None: print the user's option whether to be a premium ember
    """
    if tree.results != None:
        print tree.results
    else:
        col = tree.col
        value = tree.value
        if isinstance(value,float) or isinstance(value,int):
            if observation[col] >= value:
                classify(tree.tb,observation)
            else:
                classify(tree.fb,observation)
        else:
            if observation[col] == value:
                classify(tree.tb,observation)
            else:
                classify(tree.fb,observation)


def prune(tree, mini_entropy):
    """prune the decision tree in terms of mini_entropy 
    
    Args:
        tree (DecisionTreeNode): the root node of the tree to be processed
        mini_entropy: the threshold of entropy to decide whether to prune the tree
    
    Returns:
        DecisionTreeNode : the root of the tree after pruning 
    """
    if tree.tb.results is not None and tree.fb.results is not None:
        tb_result = []
        fb_result = []
        for k, v in tree.tb.results.items():
            tb_result += [[k]]*v
        for k, v in tree.fb.results.items():
            fb_result += [[k]]*v

        sum = len(tb_result) + len(fb_result)
        #reduce_entropy = entropy(tb_result+fb_result) - entropy(tb_result)*len(tb_result)/sum - entropy(fb_result)*len(fb_result)/sum 
        reduce_entropy = entropy(tb_result+fb_result) - (entropy(tb_result)+entropy(fb_result))/2
        if reduce_entropy < mini_entropy:
            tree.results = unique_counts(tb_result+fb_result)
            tree.tb = None
            tree.fb = None

    else:
        if tree.tb.results is None:
            prune(tree.tb, mini_entropy)
        if tree.fb.results is None:
            prune(tree.fb, mini_entropy)


def mdclassfy(tree,observation):
    """classfy observation  with some missing data
    
    Args:
        tree (TYPE): root of decision tree
        observation (TYPE): new observation to be classfied
    
    Returns:
        result that the observation to be classfied
    """
    if tree.results != None:
        return results
    col = tree.col
    if observation[col]==None: # empty field in the observation
        resutl = {}
        tb,fb = mdclassfy(tree.tb,observation),mdclassfy(tree.fb,observation)
        tb_count = sum(tb.values)
        fb_count = sum(fb.values)
        tb_fraction = tb_count/(tb_count+fb_count)
        fb_fraction = fb_count/(tb_count+fb_count)  
        for k,v in tb.items(): results[k]+=v*tb_count
        for k,v in fb.items(): results[k]+=v*fb_count
        return result
    else:
        value = tree.value
        if isinstance(value,float) or isinstance(value,int):
            if observation[col] >= value:
                return mdclassify(tree.tb,observation)
            else:
                reutrn mdclassify(tree.fb,observation)
        else:
            if observation[col] == value:
                return mdclassify(tree.tb,observation)
            else:
                return mdclassify(tree.fb,observation)



    
    
if __name__ == '__main__':
    rot = DecisionTreeNode()
    rot = build_tree(sample_data)
    print_tree(rot, '')
    
    print '\npredicting new case with decision tree'
    predict_case = ['(direct)','USA','yes',5]
    classify(rot,predict_case)
    
    print '\nafter pruning the tree'
    prune(rot,1)
    print_tree(rot,' ')
    
