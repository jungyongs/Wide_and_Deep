import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Embedding, Reshape, Flatten, Dense, Dropout, Concatenate
from keras import Model
from keras.regularizers import l2, l1_l2
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Wide&Deep.")
    parser.add_argument('--train_data', default='./data/adult.data',
                        help='Input train data path.')
    parser.add_argument('--test_data', default='./data/adult.test',
                        help='Input test data path.')
    return parser.parse_args()

def load_data(train_data, test_data):

    Columns = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

    train_df = pd.read_csv(train_data, names=Columns)
    test_df = pd.read_csv(test_data, names=Columns)
    test_df = test_df.drop(0)

    return train_df, test_df

def wide(train_df, test_df):

    train_df['IS_TRAIN'] = 1
    test_df['IS_TRAIN'] = 0
    wide_df = pd.concat([train_df,test_df])

    wide_cols = ['workclass', 'education', 'marital_status', 'occupation',
                     'relationship', 'race', 'gender', 'native_country']
    crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])
    target = 'income_label'
    categorical_cols = list(wide_df.select_dtypes(include=['object']).columns)

    # cross-product transformation
    crossed_cols_dic = cross_cols(crossed_cols)
    wide_cols += list(crossed_cols_dic.keys())

    for col_name, col_lst in crossed_cols_dic.items():
        wide_df[col_name] = wide_df[col_lst].apply(lambda x: '-'.join(x), axis=1)
    wide_df = wide_df[wide_cols + [target] + ['IS_TRAIN']]

    # one-hot encoding
    dummy_cols = [col for col in wide_cols if col in categorical_cols+list(crossed_cols_dic.keys())]
    wide_df = pd.get_dummies(wide_df, columns=dummy_cols)

    train = wide_df[wide_df.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = wide_df[wide_df.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)

    cols = [c for c in train.columns if c != target]
    X_train = train[cols].values
    X_train = np.array(X_train, dtype=float)
    y_train = train[target].values.reshape(-1, 1)

    X_test = test[cols].values
    X_test = np.array(X_test, dtype=float)
    y_test = test[target].values.reshape(-1, 1)

    return X_train, y_train, X_test, y_test

def deep(train_df, test_df):
    train_df['IS_TRAIN'] = 1
    test_df['IS_TRAIN'] = 0
    deep_df = pd.concat([train_df,test_df])

    embedding_cols = ['workclass', 'education', 'marital_status', 'occupation',
                      'relationship', 'race', 'gender', 'native_country']
    cont_cols = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
    target = 'income_label'

    deep_cols =embedding_cols + cont_cols
    deep_df = deep_df[deep_cols+[target]+['IS_TRAIN']]
    scaler = StandardScaler()
    deep_df[cont_cols] = pd.DataFrame(scaler.fit_transform(train_df[cont_cols]), columns=cont_cols)

    # Label Encoding
    deep_df, nunique = val2idx(deep_df, embedding_cols)

    train = deep_df[deep_df.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = deep_df[deep_df.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)

    # Categorical Features (Input, Embedding)
    embeddings_tensors = []
    for ec in embedding_cols:
        layer_name = ec + '_inp'
        inp = Input(shape=(1,), dtype='int64', name=layer_name)
        embd = Embedding(nunique[ec],8, input_length=1, embeddings_regularizer=l2(0.001))(inp)
        embeddings_tensors.append((inp, embd))

    # Continuous Features (Input, )
    continuous_tensors = []
    for cc in cont_cols:
        layer_name = cc + '_in'
        inp = Input(shape=(1,), dtype='float32', name=layer_name)
        bulid = Reshape((1, 1))(inp)
        continuous_tensors.append((inp, bulid))

    X_train = [train[c] for c in deep_cols]
    y_train = np.array(train[target].values).reshape(-1, 1)
    X_test = [test[c] for c in deep_cols]
    y_test = np.array(test[target].values).reshape(-1, 1)

    inp_layer =  [et[0] for et in embeddings_tensors]
    inp_layer += [ct[0] for ct in continuous_tensors]
    inp_embed =  [et[1] for et in embeddings_tensors]
    inp_embed += [ct[1] for ct in continuous_tensors]

    return X_train, y_train, X_test, y_test, inp_embed, inp_layer

def cross_cols(cols):
    crossed_cols = dict()
    colnames = ['_'.join(col) for col in cols]
    for col_name, col in zip(colnames, cols):
        crossed_cols[col_name] = col
    return crossed_cols

def val2idx(df,cols):
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o:i for i, o in enumerate(v)}
    
    for k,v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    unique_vals = dict()
    for c in cols:
        unique_vals[c] = df[c].nunique()
    
    return df, unique_vals

    
def wide_deep(train_df,test_df):
    X_train_wide, y_train_wide, X_test_wide, y_test_wide = wide(train_df, test_df)

    X_train_deep, y_train_deep, X_test_deep, y_test_deep, deep_inp_embed, deep_inp_layer = deep(train_df, test_df)

    X_train = [X_train_wide] + X_train_deep
    Y_train = y_train_deep  
    X_test = [X_test_wide] + X_test_deep
    Y_test = y_test_deep  

    # wide
    w = Input(shape=(X_train_wide.shape[1],), dtype='float32', name='wide')

    # deep
    d = Concatenate()(deep_inp_embed)
    d = Flatten()(d)
    d = Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(20, activation='relu', name='deep')(d)
    d = Dropout(0.5)(d)

    # wide + deep
    wd_inp = Concatenate()([w, d])
    wd_out = Dense(Y_train.shape[1],activation='sigmoid', name='wide_deep')(wd_inp)

    wide_deep = Model(inputs = [w,deep_inp_layer], outputs = wd_out)
    print(wide_deep.summary())
    wide_deep.compile(optimizer='Adam', loss='binary_crossentropy', metrics='accuracy')
    wide_deep.fit(X_train, Y_train, epochs=5, batch_size=128)

    results = wide_deep.evaluate(X_test, Y_test)

    print("wide and deep model accuracy:", results[1])

if __name__ == '__main__' : 
    args = parse_args()
    train = args.train_data
    test = args.test_data

    train_df, test_df = load_data(train, test)

    # logistic regression label ({<=50K : 0, >50K : 1})
    train_df['income_label'] = (
        train_df["income_bracket"].apply(lambda x: ">50K" in x)).astype(int) 
    test_df['income_label'] = (
        test_df["income_bracket"].apply(lambda x: ">50K." in x)).astype(int)
    
    wide_deep(train_df, test_df)


    
