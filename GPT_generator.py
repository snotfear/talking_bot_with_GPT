from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re

tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")

bad_word_ids = [
    [203], # \n
    [225], # weird space 1
    [28664], # weird space 2
    [13298], # weird space 3
    [206], # \r
    [49120], # html
    [25872], # http
    [3886], # amp
    [38512], # nbsp
    [10], # &
    [5436], # & (another)
    [5861], # http
    [372], # yet another line break
    [421, 4395], # МСК
    [64], # \
    [33077], # https
    [1572], # ru
    [11101], # Источник
]

def gen_fragment(context, bad_word_ids=bad_word_ids, print_debug_output=False, temperature=1.0, max_length=75, min_length=50):
    input_ids = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt").to("cpu")
    # input_ids = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt").to("cuda")
    input_ids = input_ids[:, -1700:]
    input_size = input_ids.size(1)
    output_sequences = model.generate(
        num_beams=3,
        input_ids=input_ids,
        max_length=max_length + input_size,
        min_length=min_length + input_size,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1,
        temperature=temperature,
        pad_token_id=0,
        eos_token_id=2,
        bad_words_ids=bad_word_ids,
        no_repeat_ngram_size=6
    )
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
    generated_sequence = output_sequences[0].tolist()[input_size:]
    if print_debug_output:
        for idx in generated_sequence:
            print(idx, tokenizer.decode([idx], clean_up_tokenization_spaces=True).strip())
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[: text.find("</s>")]
    text = text[: text.find("<s>")]
    first_liter = re.search(r'\w', text)
    text = text[text.find(first_liter.group(0)): ]
    if '.' in text:
      text = text[: text.rfind(".") + 1]
    elif '!' in text:
      text = text[: text.rfind("!") + 1]
    elif '?' in text:
      text = text[: text.rfind("?") + 1]
    return text

def gen_begginer(user_input, dialog_hist):
    '''
    :param user_input: ввод пользователя
    :param dialog_hist: предыдущая история
    :return: строку для затравки
    '''
    dialog = str()
    for i in dialog_hist:
        dialog += str(i)
    return str(f'{dialog}Ю.- {str(user_input)}\nБ.- ')