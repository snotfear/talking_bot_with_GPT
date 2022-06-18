import re
from razdel import tokenize

class MyTokenizer:
    def __init__(self):
        pass
    def tokenizes(self, text):
        sub_token = tokenize(text)
        return [_.text for _ in sub_token]

tokenizer = MyTokenizer()

hard_replacement = {
                      'я': 'ты',
                      'вы': 'я',
                      'меня': 'тебя',
                      'мне': 'тебе',
                      'мной': 'тобой',
                      'мною': 'тобою',

                      'тебя': 'меня',
                      'тебе': 'мне',
                      'тобой': 'мной',
                      'тобою': 'мною',

                      'вас': 'меня',
                      'вам': 'мне',
                      'вами': 'мной',

                      'по-моему': 'по-твоему',
                      'по-твоему': 'по-моему',
                      'по-вашему': 'по-моему',

                      'ваш': 'мой',
                      'ваши': 'мои',
                      'вашим': 'моим',
                      'вашими': 'моими',
                      'ваших': 'моих',
                      'вашем': 'моем',
                      'вашему': 'моему',
                      'вашей': 'моей',
                      'вашу': 'мою',
                      'ваша': 'моя',
                      'ваше': 'мое',

                      'твой': 'мой',
                      'твои': 'мои',
                      'твоим': 'моим',
                      'твоими': 'моими',
                      'твоих': 'моих',
                      'твоем': 'моем',
                      'твоему': 'моему',
                      'твоей': 'моей',
                      'твою': 'мою',
                      'твоя': 'моя',
                      'твое': 'мое',

                      'наш': 'ваш',
                                  }


def replace(s, pronounce_dict):
    if 'ё' in s:
        s = re.sub(r'ё', 'е', s)
    s = tokenizer.tokenizes(s)
    print(s)
    for i in range(len(s)):
        for key, value in pronounce_dict.items():
            if s[i] == key:
                s[i] = value
                break
            elif s[i] == value:
                s[i] = key
                break
    return ' '.join(s)


def postprocess_prepositions(s):
    s = re.sub(r'\bко тебе\b', 'к тебе', s)
    s = re.sub(r'\bобо тебе\b', 'о тебе', s)
    s = re.sub(r'\bсо тобой\b', 'с тобой', s)
    s = re.sub(r'\bво тебе\b', 'в тебе', s)

    s = re.sub(r'\bк мне\b', 'ко мне', s)
    s = re.sub(r'\bо мне\b', 'обо мне', s)
    s = re.sub(r'\bс мной\b', 'со мной', s)
    s = re.sub(r'\bв мне\b', 'во мне', s)
    return s

def total_replace(s):
    s = replace(s, pronounce_dict=hard_replacement)
    s = postprocess_prepositions(s)
    return s

text = 'Здорово! Меня зовут Илья. А тебя как?'

print(total_replace(text))