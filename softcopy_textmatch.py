import fuzzywuzzy as fuzz
from fuzzywuzzy import fuzz

import pandas as pd
import re
import string

import streamlit as st


def best_match(input_str, df, min_confidence=75):
    '''
    gets best fuzzy match of text pulled from OCR to book/author in the book crossing dataset
    output:
        1. prints a string with string of title/author as they appear in df
        2. returns ISBN of book
    -----------------------
    input_str:
        str, title/author pulled from OCR, should be 'books_all' df
    df:
        series, all cleaned and preprocessed title/authors from dataset (books_all.title_author)
    min_confidence:
        int, min fuzzy match ratio for fuzz.token_set_ratio. default set to 70%.
    '''

    match_dict = {}

    for book in df.title_author:
        match_ratio = fuzz.token_set_ratio(input_str, book)
        match_dict[match_ratio] = book

    best_match_ratio = max(match_dict.keys())
    best_match_title_author = match_dict[best_match_ratio]
    best_match_index = df[df.title_author == best_match_title_author].index[0]

    best_match_title = df.loc[best_match_index].bookTitle
    best_match_author = df.loc[best_match_index].bookAuthor
    best_match_year = df.loc[best_match_index].yearOfPublication
    best_match_publisher = df.loc[best_match_index].publisher
    best_match_isbn = df.loc[best_match_index].ISBN

    if best_match_ratio > min_confidence:
        st.markdown('* Closest match for "*{}*":  \n **{}** by **{}** with {}% confidence.'.format(input_str, best_match_title, best_match_author, best_match_ratio))
        #w.i.p.: buttons to select/deselect books from rec process
        #override = st.sidebar.checkbox(label=best_match_title, value=True)
        #if override:
        return best_match_isbn

    else:
        st.markdown('* No match for "*{}*".  \n (Closest match was "{}" with {}% confidence.)'.format(input_str, best_match_title_author, best_match_ratio))
        #override = st.sidebar.checkbox(label=best_match_title, value=False)
        #if override:
            #return best_match_isbn
