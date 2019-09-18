#!/usr/bin/env python
import sys
import PySimpleGUIQt as sg
import os
import ntpath

import pandas as pd
import numpy as np

import logging
import threading, queue
import time

from autolabeller.src.toolkit.autolabel import Preprocessor, AutoLabeller
import autolabeller.src.toolkit.autolabel as AutoLabel

from sklearn.naive_bayes import MultinomialNB

sg.SetOptions(element_padding=(0,0))

# ------ Menu Definition ------ #
menu_def = [['File', ['Open', 'Save', 'Exit', 'Properties']],
            ['Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
            ['Help', 'About...'], ]





def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def update_status(thread, status):
    ctr = 1
    while thread.is_alive():
        window.Element('output').Update(status +  str( '.....'[0:(ctr % 4)] ))
        ctr = ctr + 1
        window.Refresh()
        time.sleep(0.5)

def threadder(status, target, queue=None, *args ):
        out_queue = queue.Queue()
        x = threading.Thread(target=target, args=args)
        x.start()
        update_status(x, status)
        if queue:
            label_data = out_queue.get()
            return label_data

def read_raw_data(name, out_queue):
    raw_data = pd.read_csv(name)
    out_queue.put(raw_data)

def data_preprocess(raw_data, stopwords_path, out_queue):
    preprocessor = Preprocessor()
    corpus=raw_data['overview']
    preprocessed_corpus = preprocessor.corpus_preprocess(corpus=corpus, stopwords_path=stopwords_path)
    # Replace bigrams
    newcorpus =  preprocessor.corpus_replace_bigrams(corpus=preprocessed_corpus, min_df=50, max_df=500)
    out_queue.put(newcorpus)

def generate_labels(output_folder, newcorpus, label_words_val):

    #  Generate Recommended Labels
    window.Element('output').Update('generate topic model... ')

    # Returns a matrix of recommended words
    topic_model, dtm, best_n = AutoLabel.recommend_words(newcorpus)   #
    topic_dataframe = topic_model.show_topics(dtm=dtm, best_n=best_n, n_words=label_words_val)
    topic_dataframe.to_csv(output_folder + '/labels.csv')

def enrich(labels, corpus, raw_data, out_queue):
    autoLabeller = AutoLabeller(labels, corpus, raw_data)
    enriched_labels = autoLabeller.train(n_words=20)
    out_queue.put(autoLabeller)



def load_raw_data(output_folder, stopwords_path, label_words_val):
    output_folder = os.path.abspath(output_folder)

    filename = sg.PopupGetFile('raw data filename', no_window=True, file_types=(("CSV Files", "*.csv"),))
    print(filename)
    if filename is not None and filename != '':
        fn = path_leaf(filename)

        window.Element('output').Update('loading file: ' + str(fn))
        window.Refresh()
        # raw_data = read_raw_data(filename)

        out_queue = queue.Queue()
        # https://realpython.com/intro-to-python-threading/
        x = threading.Thread(target=read_raw_data, args=(filename, out_queue))
        x.start()
        update_status(x, 'read data')
        raw_data = out_queue.get()
        window.Element('output').Update('file loaded.')
        window.Refresh()

        # data preprocess
        window.Element('output').Update('data preprocessing...')
        window.Refresh()


        out_queue = queue.Queue()
        x = threading.Thread(target=data_preprocess, args=(raw_data, stopwords_path,out_queue))
        x.start()
        update_status(x, 'process data')
        newcorpus = out_queue.get()
        window.Element('output').Update('Data preprocess complete.')
        window.Refresh()

        # build the label dictionary
        window.Element('output').Update('build the label dictionary')
        window.Refresh()


        x = threading.Thread(target=generate_labels, args=(output_folder, newcorpus, label_words_val))
        x.start()
        update_status(x, 'generate labels')
        window.Element('output').Update('Label dictionary written to labels.csv. Create labels file then load. ')
        window.Refresh()

        return raw_data, raw_data['overview']
    else:
        return None, None


def get_label_file():
    filename = sg.PopupGetFile('label data filename', no_window=True, file_types=(("CSV Files", "*.csv"),))

    if filename is not None:
        fn = path_leaf(filename)

        window.Element('output').Update('loading label file: ' + str(fn))
        window.Refresh()

        out_queue = queue.Queue()
        x = threading.Thread(target=read_raw_data, args=(filename, out_queue))
        x.start()
        update_status(x, 'read data')
        label_data = out_queue.get()
        window.Element('output').Update('file loaded.')
        window.Refresh()
        return label_data


def enrich_labels(raw_data, corpus):
    label_data = get_label_file()

    window.Element('output').Update('enrich labels')
    window.Refresh()

    out_queue = queue.Queue()
    x = threading.Thread(target=enrich, args=(label_data, corpus, raw_data, out_queue))
    x.start()
    update_status(x, 'read data')
    enriched_labels = out_queue.get()
    window.Element('output').Update('file loaded.')
    window.Refresh()
    return enriched_labels


def model_application(enriched_labels):
    mnb = MultinomialNB()
    ypred = enriched_labels.apply(mnb)


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_settings(min_df_val, max_df_val, label_words_val, folder_val, stopwords_path):
    layout2 = [
        [sg.Text('Parameter Settings', font=("Ariel", 12))],
        [sg.Text('min_df', size=(15, 1), font=("Ariel", 12)), sg.InputText(str(min_df_val), font=("Ariel", 12), key='min_df_val') ],
        [sg.Text('max_df', size=(15, 1), font=("Ariel", 12)), sg.InputText(str(max_df_val), font=("Ariel", 12), key='max_df_val')],
        [sg.Text('Number of Label Words', size=(15, 1), font=("Ariel", 12)), sg.InputText(str(label_words_val), font=("Ariel", 12), key='label_words_val')],
        [sg.Txt('Output Folder:', size=(10, 1), font=("Ariel", 12)), sg.InputText(str(folder_val), size=(30, 1), font=("Ariel", 12), key='folder_val'), sg.FolderBrowse(font=("Ariel", 12))],
        [sg.Txt('Stopwords file:', size=(10, 1), font=("Ariel", 12)), sg.InputText(str(stopwords_path), size=(30, 1), font=("Ariel", 12), key='stopwords_path'), sg.FileBrowse(font=("Ariel", 12))],
        [sg.Submit(font=("Ariel", 12)), sg.Cancel(font=("Ariel", 12))]
    ]

    settingswdw = sg.Window('Settings', grab_anywhere=False, resizable=False ).Layout(layout2)

    #settingswdw.Refresh()

    while True:  # Event Loop
        event, values = settingswdw.Read()
        print(event, values)
        if event is None or event == 'Cancel':
            print('None or Exit event')
            break
        elif event == 'Submit':

            if  not RepresentsInt(values['min_df_val']) or not RepresentsInt(values['max_df_val']) or not RepresentsInt(values['label_words_val']):
                sg.PopupError('Values must be integers, please correct.')
            elif not os.path.exists(folder_val) or not os.path.isdir(folder_val):
                sg.PopupError('Folder not valid, please correct.')
            elif not os.path.exists(stopwords_path) or not os.path.isfile(stopwords_path):
                sg.PopupError('Stopwords file not valid, please correct.')
            else:
                min_df_val = values['min_df_val']
                max_df_val = values['max_df_val']
                label_words_val = values['label_words_val']
                folder_val = values['folder_val']
                stopwords_path = values['stopwords_path']
                #print('min_df_val: ' + min_df_val + '  max_df_val:' + max_df_val + '  label_words_val:' + label_words_val + '  output_folder_val: ' + folder_val)

                break


    print( 'min_df_val: ' + min_df_val + '  max_df_val:' + max_df_val + '  label_words_val:' + label_words_val + '  output_folder_val: ' + folder_val + '  stopwords_path: ' + stopwords_path)
    settingswdw.Close()
    return min_df_val, max_df_val, label_words_val, folder_val, stopwords_path




layout = [[sg.Menu(menu_def, tearoff=True)],
          [sg.Txt('Status:', size=(10, 1), font=("Ariel", 16)), sg.Txt('Load raw data file.', size=(30,2), font=("Ariel", 16), key='output') ],
          [sg.Button('Load Data', size=(15, 1), font=("Ariel", 16)), sg.Button('Settings', size=(15, 1), font=("Ariel", 16)),  sg.Button('Exit', size=(12, 1), font=("Ariel", 16))]]

# should be : [sg.FileBrowse(), sg.Exit()]]

window = sg.Window('AIMS Auto Labeller', grab_anywhere=False, resizable=True).Layout(layout)

# Perhaps replace with a simple state machine
currentstate = 'processraw'

# parameters
min_df_val = 3
max_df_val = 300
label_words_val = 20
folder_val = os.path.expanduser('~')
stopwords_path = '/Users/mmanning/Dev/code/aims/autolabeller/data/stopwords.csv'

while True:  # Event Loop
    corpus = ''
    raw_data = ''

    event, values = window.Read()
    print(event, values)
    if event is None or event == 'Exit':
        print('None or Exit event')
        break

    if event == 'Load Data':
        if currentstate == 'processraw':
            raw_data, corpus = load_raw_data(folder_val, stopwords_path, label_words_val)
            print(raw_data)
            print('----------')
            print(corpus)

            if raw_data is not None:
                currentstate == 'buildmodel'

        elif currentstate == 'buildmodel':
            enriched_labels= enrich_labels(raw_data, corpus)
            model_application(enriched_labels)


    if event == 'Settings':
        min_df_val, max_df_val, label_words_val, folder_val, stopwords_path = get_settings(min_df_val, max_df_val, label_words_val, folder_val, stopwords_path)


window.Close()
