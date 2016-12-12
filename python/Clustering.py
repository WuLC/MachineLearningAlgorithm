# -*- coding: utf-8 -*-
"""
"""
# @Author: WuLC
# @Date:   2016-12-11 22:04:08
# @Last modified by:   WuLC
# @Last Modified time: 2016-12-12 22:50:55
# @Email: liangchaowu5@gmail.com

# @Referer: chaper 3 of the book 《programming-collective-intelligence》

import os
import io
import re
import feedparser
from collections import defaultdict


def extract_words(content):
    """extract words from content, just deal with English here
    
    Args:
        content (str): content to be extracted words
    
    Returns:
        list: a list of words that are extracted from content 
    """
    txt = re.compile(r'<[^>]+>').sub('',content)  # remove tag in the form of <XXXX>
    words = re.compile(r'[^A-Z^a-z]+').split(content) # split words by all non-alpha characters
    return [word.lower() for word in words if word != ''] # turn all words into lowercase


def parse_rss(target_url):
    """parse a url which is the rss of a blog
    
    Args:
        target_url (str): url of the rss
    
    Returns:
        (title,word_count): title of the blog and how many times that each word appears in the blog
    """
    rss = feedparser.parse(target_url)
    word_count = defaultdict(int)
    for entry in rss.entries:  # traverse all passages of the blog
        if 'summary' in entry:
            summary = entry.summary # su
        else:
            summary = entry.description
        words = extract_words(entry.title+' '+summary)
        for word in words:
            word_count[word] += 1
    return rss.feed.get('title', 'empty title'), word_count  # title can be empty sometimes


def get_content_from_feedlist(feed_list, data_file):
    """extract content from every rss in feedlist, store them in the data_file
       reduce the number of total words by selecting those words that appear within maximum and minimum percentage
    Args:
        feedlist (str): path of the feedlist file, each row represent a rss
        data_file (str): path of the data file
    
    Returns:
        None
    """
    word_appear_count = defaultdict(int) # count thow many blogs does a word appear in
    blog_word_count = {} # words of each blog
    empty_title_count = 0
    for rss_url in file(feed_list):
        title, wc = parse_rss(rss_url.strip())
        if title == 'empty title':  # cannot get title of some rss
            empty_title_count += 1
            title = title+' %s'%empty_title_count
        blog_word_count[title] = wc
        for word, count in wc.items():
            word_appear_count[word] += 1

    # caculate the appearing percentage of each word
    # record those words that appear within maximum and minimum percentage 
    minimum, maximum = 0.1, 0.5
    word_list = []
    total_blog = len(blog_word_count)
    for word, count in word_appear_count.items():
        if minimum <= count*1.0/total_blog <= maximum:
            word_list.append(word)

    # write data into data_file 
    with io.open(data_file, mode = 'w', encoding = 'utf8') as wf:
        wf.write('Blog'.decode('utf8'))
        for word in word_list:
            wf.write(('\t%s'%word).decode('utf8'))
        wf.write('\n'.decode('utf8'))
        # words of each blog
        for blog_title, blog_words in blog_word_count.items():
            wf.write(blog_title.decode('utf8'))
            for word in word_list:
                if word in blog_words:
                    wf.write(('\t%s'%blog_words[word]).decode('utf8'))
                else:
                    wf.write(('\t'+'0').decode('utf8'))
            wf.write('\n'.decode('utf8'))


if __name__ == '__main__':
    feed_list = 'Clustering_data/feedlist.txt'
    data_file = 'Clustering_data/data'
    if not os.path.exists(data_file):
        get_content_from_feedlist(feed_list, data_file)
    

    
