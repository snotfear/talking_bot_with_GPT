import random
import numpy as np
import telebot
from navec import Navec
from numpy.linalg import norm
from razdel import tokenize
import os
import time
from random import randrange
from spellchecker import SpellChecker
import ruBERT_synonymy_and_relevancy
import replace_pronounce
import torch
import GPT_interpretator
import GPT_generator
import GPT_generation2
import json

spell = SpellChecker(language='ru')

'''Загрузим готовый эмбендинг Navec от проекта Natasha'''
path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question_dict = dict()
temp_list = []

'''Токен и запуск бота'''
token = '5069703088:AAEA-gfnn-tn9yy6i616eNtiV4sEUJa3x4E'
bot = telebot.TeleBot(token)

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_path = 'ruBert-base'
bert_tokenizer = ruBERT_synonymy_and_relevancy.transformers.BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
bert_model = ruBERT_synonymy_and_relevancy.transformers.BertModel.from_pretrained(bert_path)
bert_model.to(device)
bert_model.eval()

model_interpr = GPT_interpretator.RugptBase()
model_interpr.load_from_path('inkoziev/rugpt_interpreter')

with open(os.path.join(bert_path, 'pq_relevancy_rubert_model.cfg'), 'r') as f:
    cfg = json.load(f)
    relevancy_detector = ruBERT_synonymy_and_relevancy.RubertRelevancyDetector(device=device,**cfg)
    relevancy_detector.load_weights(os.path.join(bert_path, 'pq_relevancy_rubert_model.pt'))
    relevancy_detector.bert_model = bert_model
    relevancy_detector.bert_tokenizer = bert_tokenizer

with open('facts.txt', 'r') as facts:
    memory_phrases = []
    for ind, i in enumerate(facts.readlines()):
        i = i[:-1]
        if len(i) > 2:
            if '|' in i:
                sep_i = i.split(' | ')
                i = random.choice(sep_i)
            memory_phrases.append((i, ind))

'''Загрузим файл с вопрос-ответами'''
with open('question-answer.txt', 'r', encoding='utf-8') as file:
    temp_data = file.read().splitlines()
    '''И разобъём строки по табуляции (ключи) и точке-с-запятой (значения)
    Ключами будут известные вопросы, значения - варианты ответов на них {'вопрос':[ответ_1, ответ_2...]}
    '''
    for line in temp_data:
        temp_list.append(line.split('\t'))
        for keys_and_value in temp_list:
            question_dict[keys_and_value[0]] = keys_and_value[1].split(';')
        temp_list = []


def write_dialogs(uid, user_phrase, bot_answer):
    '''
    :param uid: id пользователя
    :param user_phrase: Фраза пользователя
    :param bot_answer: ответ бота
    :return: Проводится запись в файл диалога
    '''
    with open(f'{uid}.txt', 'a', encoding='utf-8') as file:
        file.write(f'{user_phrase}\t{bot_answer}\n')

def read_dialogs(uid, len_memories, flag: str):
    '''
    :param uid: id пользователя
    :param len_memories: сколько реплик диалога помнить?
    :return: возвращает последние 10 строк диалога (5 вопросов, 5 ответов)
    '''
    dialogs = []
    if os.path.isfile(f'{uid}.txt'):
        with open(f'{uid}.txt', 'r', encoding='utf-8') as auf:
            tmp = auf.readlines()
            if len(tmp) <= len_memories:
                max_limit = 0
            else:
                max_limit = len(tmp)-len_memories
            for i in range(max_limit, len(tmp)):
                pair = tmp[i].split('\t')
                user_str = str('Ю.- ') + pair[0]
                bot_str = str('Б.- ') + pair[1]
                user_str = user_str.strip('\n')
                bot_str = bot_str.strip('\n')
                if flag == 'interpretator':
                    dialogs.append(user_str)
                    dialogs.append(bot_str)
                elif flag == 'generator':
                    dialogs.append(f'{user_str}\n{bot_str}')
            return dialogs
    return []


def generate_list_of_question(dict_with_q_a):
    '''
    :param dict_with_q_a: Словарь с вопросами и ответами
    :return: Список вопросов, будет нужен для функции ранжирования и возврата нужного ответа
    '''
    return [i for i in dict_with_q_a.keys()]


'''для разбивки на токены применяется функция tokenize, обернём её в класс для удобства'''
class MyTokenizer:
    def __init__(self):
        pass
    def tokenizes(self, text):
        sub_token = tokenize(text)
        return [_.text for _ in sub_token]


'''и сразу создадим объект класса, который будем использовать при токенизации вопрос-ответов'''
tokenizer_razdel = MyTokenizer()


def question_to_vec(question, embeddings, tokenizer, dim=300):
    '''
    Функция перевода предложения в вектор
    :param question: строка предложения
    :param embeddings: эмбеддинг (navec)
    :param tokenizer: токенизатор (razdel)
    :return: векторное представления всего вопроса
    '''
    question = question.lower()
    summ = 0
    count = 0
    tokkens = tokenizer.tokenizes(question)
    for i in tokkens:
        if i in embeddings:
            summ += embeddings[i]
            count += 1
    if count == 0:
        return np.array([10 for i in range(dim)])
    return summ / count


def rank_candidates(question, candidates_vec, embeddings, tokenizer, dim=300):
    '''
    Функция ранжирования из списка вопросов
    :param question: строка вопроса
    :param candidates_vec: список известных вопросов в виде векторных представлений
    :param embeddings: эмбеддинг
    :param tokenizer: токенизатор
    :return: вернёт индекс наиболее близкого вопроса из базы вопросов
    '''

    data = []
    q_vec = question_to_vec(question, embeddings, tokenizer_razdel, dim)
    for ind, candidate_vec in enumerate(candidates_vec):
        cos_dist = 1 - cos_simm(q_vec, candidate_vec)
        data.append([cos_dist, ind, candidate_vec])
    data.sort(key=lambda x: x[0], reverse=False)

    return data[0][:2]


def cos_simm(v1, v2):
    '''
    :param v1 и v2: массивы - векторные представления слов или предложений
    :return: Косинусную разность, float
    '''
    return np.array(v1 @ v2.T / norm(v1) / norm(v2))


def random_for_answer(index_q):
    '''
    Функция для случайного выбора ответа из списка представленных
    :param index_q: индекс вопроса, который ближе всех по смыслу к заданному пользователем
    :return: случайный индекс для ответа
    '''
    random_int = randrange(0, len(question_dict[question_list[index_q]]))
    return random_int


'''Преобразуем известные вопросы в список с векторными представлениям для более быстрого сравнения'''
vectors_of_question = []
question_list = generate_list_of_question(question_dict)
for question in question_list:
    vectors_of_question.append(question_to_vec(question=question, embeddings=navec, tokenizer=tokenizer_razdel))

'''Словарь служит для контроля id и отработки исправлений'''
dialog_history = dict()


def control_history(dialog, message, answer, cos_dist, id):
    if len(dialog[id]) > 1:
        dialog[id].pop(0)
        dialog[id].append([message, answer, cos_dist])
    else:
        dialog[id].append([message, answer, cos_dist])


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет! Я Лита, а как тебя зовут?")


@bot.message_handler(content_types=['text'])
def send_text(message):
    id = message.chat.id

    '''Если пользователь новый, то его id добавляется в словарь'''
    if id not in dialog_history:
        dialog_history[id] = [['', '', ''], ['', '', '']]

    if dialog_history[id][1][1] == 'Напиши как бы ответил ты, пожалуйста':
        with open('bad_answer.txt', 'a', encoding='utf-8') as file:
            file.write(f'{dialog_history[id][0][0]}\t{message.text}\n')
        bot.send_message(message.chat.id, text='Спасибо, я запомню твой твой ответ!')
        control_history(dialog_history, dialog_history[id][0][0], message.text, 0, id=id)
    else:
        '''Основная ветка'''
        user_q = str(message.text.lower())
        tokkens = tokenizer_razdel.tokenizes(user_q)
        user_q = ' '.join([spell.correction(i) for i in tokkens])
        rank = rank_candidates(question=user_q, candidates_vec=vectors_of_question, embeddings=navec,
                               tokenizer=tokenizer_razdel)
        user_q = 'Ю.' + user_q
        if rank[1]: #> 0.1:  # Если между вопросом пользователя и вопросом в базе косинусное расстояние слишком большое, то
            # пробуем генерацию с помощью ruGPT3
            processed_chitchat_contexts = set()
            dialog_hist = read_dialogs(uid=id, len_memories=2, flag='interpretator') # Загружаем диалог для интерпритации
            dialog_hist.append(user_q)
            print(dialog_hist, 'история диалога + запрос пользователя')


            context = GPT_interpretator.constuct_interpreter_contexts(dialog_hist[-2:])
            print(context, 'контекст для интерпритаций')
            interpretation = GPT_interpretator.generate_interpretations(model=model_interpr,
                                                                        context_replies=context,
                                                                        num_return_sequences=1)[0].lower()# Пока так, потом прикрутим выборку наиболее удачной интерпритации
            interpretation_r_p = replace_pronounce.total_replace(interpretation) #смена местоимений для определения релевантностей
            print(interpretation, 'интерпретация')

            premises0, rels0 = relevancy_detector.get_most_relevant(interpretation_r_p, memory_phrases, nb_results=1)
            premises = []
            rels = []
            print(premises0, rels0)

            for premise, premise_rel in zip(premises0, rels0):
                if premise_rel >= 0.8:
                    premises.append(premise)
                    rels.append(premise_rel)
                    premise_fact = premises[0]
                    print(premise_fact, 'выбранный факт')
            if premises:
                print(dialog_hist[-1:], 'диалог в генерацию')
                answer = GPT_generation2.generate_pqa_reply(dialog=dialog_hist[-1:], interpretation=interpretation_r_p,
                                                        processed_chitchat_contexts=processed_chitchat_contexts,
                                                        premise_facts=[premise_fact])
            else:
                answer = GPT_generation2.generative_model.generate_confabulations([interpretation], num_return_sequences=1)
            answer = ''.join(answer)
            answer_interpr = GPT_interpretator.generate_interpretations(model=model_interpr,
                                                                        context_replies=[interpretation, answer], num_return_sequences=1)


            # dialog_hist = read_dialogs(uid=id, len_memories=4, flag='generator')# Загружаем диалог для генерации
            # beginning = GPT_generator.gen_begginer(user_input=user_q, dialog_hist=dialog_hist)
            # answer = GPT_generator.gen_fragment(beginning, temperature=1, max_length=30, min_length=2)
            write_dialogs(uid=id, user_phrase=''.join(interpretation), bot_answer=''.join(answer_interpr))
            bot.send_message(message.from_user.id, answer, reply_markup=get_keyboard())
            control_history(dialog_history, user_q, dialog_history[id][0][0], 0, id=id)
        else:
            # Если же косинусное расстояние меньше, то ответ берем из базы ответов
            answer = question_dict[question_list[rank[1]]][random_for_answer(rank[1])]
            time.sleep(3)
            bot.send_message(message.chat.id, answer, reply_markup=get_keyboard())
            control_history(dialog_history, user_q, answer, rank[0], id=id)
            write_dialogs(uid=id, user_phrase=message.text, bot_answer=answer)


def get_keyboard():
    '''Используется в качестве reply_markup при отправке сообщения'''
    buttons = [
        telebot.types.InlineKeyboardButton(text='Плохой ответ!', callback_data='bad_answer')
    ]
    keyboard_gen = telebot.types.InlineKeyboardMarkup(row_width=1)
    keyboard_gen.add(*buttons)
    return keyboard_gen


@bot.callback_query_handler(func=lambda call: True)
def bad_answer(call):
    '''Описывает действия при нажатии на кнопку'''
    id = call.message.chat.id
    if call.data == 'bad_answer':
        bot.send_message(call.message.chat.id, text='Напиши как бы ответил ты, пожалуйста')
        control_history(dialog=dialog_history, message='', answer='Напиши как бы ответил ты, пожалуйста', cos_dist='',
                        id=id)


bot.infinity_polling()
