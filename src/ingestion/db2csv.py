"""Created on Mon Dec 18 13:44:45 2023.

@author: aslan

This module contains converter from Microsoft SQL Server to CSV
"""

import csv
import os
import numpy as np
import pandas as pd
import pyodbc
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

def save_tk(tag_size=None, chunk_size=100, filename='tk.csv', 
            tag_filename='tk-tag.csv', min_hit_count=1, max_chunks=None):
    """Convert terapikulubu posts and tags into CSV format."""
   
    user, password = os.environ['TK_CREDENTIALS'].split()
    
    conn = pyodbc.connect(Driver='SQL Server', host='31.220.81.76',
                          database='AVS', user=user,
                          password=password)
    file = open(filename, 'w', encoding='utf-8', newline='')  
    with conn, file:
        file_writer = csv.writer(file, lineterminator=os.linesep)
        file_writer.writerow(['body_text', 'tags'])
                               
        cur = conn.cursor()
        
        cur.execute("""SELECT COUNT(*) FROM posts""")
        total_rows = cur.fetchone()[0]
                    
        cur.execute("""SELECT tag_id, name, hit_count FROM tags
                    WHERE type=1 and hit_count >= ? ORDER BY hit_count DESC""", (min_hit_count, ))
        rows = cur.fetchall()
        tag_ids = [row[0] for row in rows]
        tag_ids_set = set(tag_ids)
        tag_names =  [row[1] for row in rows]
        tag_hit_counts =  [row[2] for row in rows]
                   
        tag_file = open(tag_filename, 'w', encoding='utf-8', newline='')    
        tag_file_writer = csv.writer(tag_file, lineterminator=os.linesep)
        tag_file_writer.writerow(['tag_id', 'tag_name', 'hit_count'])
        tag_file_writer.writerows(zip(tag_ids, tag_names, tag_hit_counts))
        tag_file.close()
                                   
        offset = 0
        file_row = 0
        while True:           
            cur.execute(f"""SELECT post_id, header, body_text FROM posts
                    WHERE is_active = 'Y' AND is_deleted = 0 AND visibility 
                    IN (0, 1, 2, 3, 4, 5, 6, 7, 8)
                    ORDER BY post_id ASC OFFSET ? ROWS
                    FETCH NEXT {chunk_size} ROWS ONLY""", offset)
            if not (rows := cur.fetchall()):
                break
     
            for index, (post_id, header, body_text) in enumerate(rows):
                cur.execute("""SELECT tag_id FROM post_tags
                    WHERE is_active='Y' AND post_id = ? 
                    ORDER BY "index" ASC;""", post_id)
                if rows := cur.fetchall():
                    post_tags = list(map(lambda row: row[0], filter(lambda row: row[0] in tag_ids_set, rows)))
                                                                                               
                    bs = BeautifulSoup(body_text, features="lxml")
                    body_text = bs.get_text(' ')
                    
                    file_writer.writerow([body_text] + post_tags)  
                    file_row += 1
                  
            offset += chunk_size
            
            print(f'{offset} of {total_rows}, {file_row} rows written')
            
            if max_chunks is not None and offset // chunk_size >= max_chunks:
                break
                 
    print(f'{file_row} row(s) written...')
    
def read_tk(filename='tk.csv', tag_filename='tk-tag.csv'):
    tag_dict = {} 

    tags_df = pd.read_csv(tag_filename)
    for i in range(len(tags_df)):
        tag_id, tag_name, hit_count = tags_df.iloc[i]
        tag_dict[tag_id] = i, tag_name

    posts_tags_df = pd.DataFrame(columns=['body_text'] + list(tags_df['tag_name']))
    with open(filename, 'r', encoding='utf-8', newline='') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)        # pass header
        for index, row in enumerate(csv_reader):
            tags_col = [tag_dict[int(tag)][0] for tag in row[1:]]
            tags_array = np.zeros(len(tags_df), dtype=np.uint8)
            tags_array[tags_col] = 1
            posts_tags_df.loc[index] = [row[0], *tags_array]
            
    return tag_dict, tags_df, posts_tags_df
            
if __name__ == '__main__':
    try:
        save_tk(min_hit_count=10, max_chunks=None)
        tag_dict, tags_df, posts_tags_df = read_tk()
        posts_tags_df.to_csv('tk_df.csv', index=False)
        
        nltk.download('stopwords')
        sw = stopwords.words('turkish')
        cv = CountVectorizer(dtype=np.uint8, stop_words=sw)
        dataset_x = cv.fit_transform(posts_tags_df['body_text']).todense()
        dataset_y = posts_tags_df.iloc[:, 1:].to_numpy()
    except OSError as e:
        print(e)



