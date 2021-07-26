"""
Stuff I ended up not using, but don't want to delete quite yet
"""


def get_feats(texts, vec_params, preproc=False):
    if preproc:
        texts = preprocess_tweets(texts)
    feats = []
    word_Tfidf_vec = TfidfVectorizer(analyzer='word',
                                     tokenizer=TweetTokenizer().tokenize,
                                     ngram_range=vec_params['word_range'],
                                     min_df=vec_params['word_min'], max_df=vec_params["word_max"],
                                     use_idf=True,
                                     sublinear_tf=True)
    feats.append(('wgram', word_Tfidf_vec))

    char_Tfidf_vec = TfidfVectorizer(analyzer='char',
                                     tokenizer=TweetTokenizer().tokenize,
                                     ngram_range=vec_params['char_range'],
                                     min_df=vec_params['char_min'], max_df=vec_params["char_max"],
                                     use_idf=True,
                                     sublinear_tf=True)
    feats.append(('cgram', char_Tfidf_vec))
    vectorizer = Pipeline([("feats", FeatureUnion(feats))])

    return vectorizer.fit_transform(texts)


def create_xy(DF, vec_params, concat=True, preproc=False):
    y = DF['class']
    if concat:
        return train_test_split(get_feats(DF['concat'], vec_params, preproc), y, test_size=0.20, random_state=42,
                                shuffle=False)
    else:
        return train_test_split(vec_per_tweet(DF, vec_params, preproc), y, test_size=0.20, random_state=42,
                                shuffle=False)


def vec_per_tweet(DF, vec_params):
    indv_tweets = DF.drop(['lang', 'class', 'authorID'], axis=1)
    data_for_model = []
    for column in indv_tweets:
        data_for_model.append(get_feats(indv_tweets[column], vec_params))

    return hstack(data_for_model)


def get_feats2(DF, vec_params):
    word_Tfidf_vec = TfidfVectorizer(analyzer='word',
                                     tokenizer=TweetTokenizer().tokenize,
                                     ngram_range=vec_params['word_range'],
                                     min_df=vec_params['word_min'], max_df=vec_params["word_max"],
                                     use_idf=True,
                                     sublinear_tf=True)
    char_Tfidf_vec = TfidfVectorizer(analyzer='char',
                                     tokenizer=TweetTokenizer().tokenize,
                                     ngram_range=vec_params['char_range'],
                                     min_df=vec_params['char_min'], max_df=vec_params["char_max"],
                                     use_idf=True,
                                     sublinear_tf=True)

    column_transformer = ColumnTransformer(
        [('tfidf_word', word_Tfidf_vec, 'concat_1'),
         ('tfidf_char', char_Tfidf_vec, 'concat_2')],
        remainder='passthrough')

    return column_transformer.fit_transform(DF)
