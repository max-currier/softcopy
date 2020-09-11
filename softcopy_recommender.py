# data munging
import numpy as np
import pandas as pd
from scipy.linalg import svd
import streamlit as st

def get_book_details(ISBN_, books_all_df):
    '''
    takes ISBN and returns details about book (title, author, year, publisher)
    in for form of a dictionary.
    ------------------------------------------------
    ISBN_:
        str, ISBN for a book

    books_df:
        dataframe, books that are included in ratings matrix that were not removed during data cleaning

    book_all_df:
        dataframe, all books in a dataframe, including those that were removed during cleaning.
    '''

    try:
        if ISBN_ is not None:
            detail_dict = {}
            if books_all_df[books_all_df.ISBN == ISBN_] is not None:
                index = books_all_df[books_all_df['ISBN']==ISBN_].index[0]
                detail_dict['title'] = books_all_df.loc[index].bookTitle
                detail_dict['author'] = books_all_df.loc[index].bookAuthor
                detail_dict['year'] = books_all_df.loc[index].yearOfPublication
                detail_dict['publisher'] = books_all_df.loc[index].publisher
                detail_dict['image_s'] = books_all_df.loc[index].imageUrlS
                detail_dict['image_m'] = books_all_df.loc[index].imageUrlM
                detail_dict['image_l'] = books_all_df.loc[index].imageUrlL

                return detail_dict
        else:
            pass

    except IndexError:
        st.write("ISBN {} not found in books dataset.".format(ISBN_))
        pass


def ISBN_to_title(ISBN_, books_all_df):
    '''
    takes ISBN and returns the book title and author
    from list of all ISBNs (including those not in the cleaned books df)
    ------------------------------------------------
    ISBN_:
        str, ISBN for a book

    books_df:
        dataframe, books that are included in ratings matrix that were not removed during data cleaning

    book_all_df:
        dataframe, all books in a dataframe, including those that were removed during cleaning.
    '''
    try:
        if ISBN_ is not None:
            if books_all_df[books_all_df.ISBN == ISBN_] is not None:
                index = books_all_df[books_all_df['ISBN']==ISBN_].index[0]
                title = books_all_df.loc[index].title_author # change to iloc if error
                return title
        else:
            pass

    except IndexError:
        st.write("ISBN {} not found in books dataset.".format(ISBN_))
        pass


def drop_none(list_):
    return [x for x in list_ if x]


def get_recommends(ISBN_val, VT, ISBNs, num_recom=1):
    '''
    Takes book ISBN value and returns n recommended book ISBNs

    ISBN_val:
        str, ISBN numbers of books to base recommendations off of

    VT:
        matrix, product-feature matrix generated from SVD

    num_recom:
        int, number or recs to return

    ISBNs:
        object, list of columns from user-item matrix
        aka user_item_mat.columns
    '''
    recs = []

    if ISBN_val is None:
        pass

    elif ISBNs.ISBN.isin([ISBN_val]).any():
        # converts ISBN to index
        itemID = ISBNs[ISBNs.ISBN.isin([ISBN_val])].index[0] # converts ISBN to index
        for item in range(VT.T.shape[0]):
            if item != itemID:
                recs.append([item, np.dot(VT.T[itemID], VT.T[item])])

        final_rec_index = [i[0] for i in sorted(recs, key=lambda x: x[1], reverse=True)]
        final_rec_isbn = [ISBNs.ISBN.loc[i] for i in final_rec_index] # takes index and returns ISBN of book
        final = drop_none(final_rec_isbn[:num_recom])

        return final

    else:
        #st.write('{} not in recommendation matrix'.format(ISBN_val))
        pass


def get_recommends_list(ISBN_list, VT, ISBNs,  n=1):
    '''
    takes list of ISBNs and returns recommendations and ISBNs of books used as basis for those recs
    ------------------------------------------------
    ISBN_list:
        list, ISBN numbers of books to base recommendations off of

    VT:
        matrix, product-feature matrix generated from SVD

    n:
        int, number or recs to return for each ISBN

    ISBNs:
        object, list of columns from user-item matrix
        aka user_item_mat.columns
    '''
    recs = []
    basis = []
    for ISBN_val in ISBN_list:
        rec = get_recommends(ISBN_val, VT, ISBNs, num_recom=n)
        if rec:
            recs.append(rec)
            basis.append(ISBN_val)
    flat_recs = [i for sub in recs for i in sub]
    return flat_recs, basis
