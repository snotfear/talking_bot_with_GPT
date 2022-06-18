import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import re


class RugptBase:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.beam_k = 10
        self.beam_p = 0.9

    def load_from_path(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def generate_output_from_prompt(self, prompt_text, num_return_sequences, temperature=1.0):
        repetition_penalty = 1.0
        stop_token = "</s>"
        length = 100

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=self.beam_k,
            top_p=self.beam_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=0
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = set()
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            if stop_token in text:
                text = text[: text.find(stop_token)]
            print(text, 'text')
            # total_sequence = text[text.find('#') + 1:].strip()  # отрезаем промт
            total_sequence = text[len(prompt_text)-1:].strip()  # отрезаем промт
            print(total_sequence, 'текст, обрезанный')
            if total_sequence:
                generated_sequences.add(total_sequence)

        return list(generated_sequences)

class RugptChitchat(RugptBase):
    def __init__(self):
        super(RugptChitchat, self).__init__()
        self.beam_k = 50
        self.beam_p = 0.9
        self.temperature = 1.0

    def load(self, model_path):
        self.load_from_path(model_path)

    def generate_chitchat(self, context_replies, num_return_sequences):
        outputs = set()
        print(context_replies, 'контекст внутри генерации')
        input_dialog = []
        for r in context_replies:
            if r.startswith('[') or r.startswith('{'):
                # Специальные метки в квадратных скобочках
                input_dialog.append(r)
            elif r.startswith('-'):
                # Обычные реплики, в начале которых уже стоит тире
                input_dialog.append(r)
            else:
                # Обычные реплики без начального "- "
                input_dialog.append('- ' + r)

        prompt_text = '<s>{chitchat}\n' + '\n'.join(input_dialog) + '\n'
        raw_outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences, temperature=self.temperature)
        print(raw_outputs, 'сырой выход модели')
        for o in raw_outputs:
            lines = o.split('\n')
            line1 = lines[0].strip()
            if line1.startswith('-'):
                line1 = line1[1:-1].strip()
            outputs.add(line1)

        return list(outputs)

    def score_dialogues(self, dialogues):
        # из-за разной длины текстов придется выполнять вычисления по 1 тексту за раз :(
        scores = []
        for dialog in dialogues:
            encoded_text = self.tokenizer.encode('<s>{chitchat}\n' + '\n'.join(dialog))
            t = torch.tensor(encoded_text, dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.no_grad():
                loss = self.model(t, labels=t)
            #perplexity = math.exp(loss[0].item())
            score = loss[0].item()
            scores.append(math.exp(-score))

        return scores

    def generate_autoquestions(self, context_replies, num_return_sequences):
        outputs = set()

        input_dialog = []
        for r in context_replies:
            if r.startswith('[') or r.startswith('{'):
                input_dialog.append(r)
            elif r.startswith('-'):
                # Обычные реплики, в начале которых уже стоит тире
                input_dialog.append(r)
            else:
                # Обычные реплики без начального "- "
                input_dialog.append('- ' + r)

        prompt_text = '<s>{autoquestion}\n' + '\n'.join(input_dialog) + '\n'
        raw_outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences, temperature=self.temperature)
        for o in raw_outputs:
            lines = o.split('\n')
            line1 = lines[0].strip()
            if line1.startswith('-'):
                line1 = line1[1:-1].strip()
            outputs.add(line1)

        return list(outputs)

    def generate_confabulations(self, context_replies, num_return_sequences):
        outputs = set()

        input_dialog = []
        for r in context_replies:
            if r.startswith('[') or r.startswith('{'):
                input_dialog.append(r)
            elif r.startswith('-'):
                # Обычные реплики, в начале которых уже стоит тире
                input_dialog.append(r)
            else:
                # Обычные реплики без начального "- "
                input_dialog.append('- ' + r)

        prompt_text = '<s>{confabulation}\n' + '\n'.join(input_dialog) + '\n'
        raw_outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences, temperature=self.temperature)
        for o in raw_outputs:
            lines = o.split('\n')
            line1 = lines[0].strip()
            if line1.startswith('-'):
                line1 = line1[1:-1].strip() # [1:-1] - убираем дефис и знак переноса строки
            outputs.add(line1)

        return list(outputs)

    def generate_interpretations(self, context_replies, num_return_sequences):
        input_context = []
        for r in context_replies:
            if r.startswith('[') or r.startswith('{'):
                input_context.append(r)
            elif r.startswith('-'):
                # Обычные реплики, в начале которых уже стоит тире
                input_context.append(r)
            else:
                # Обычные реплики без начального "- "
                input_context.append('- ' + r)

        prompt_text = '<s>' + '\n'.join(input_context) + ' #'
        outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences, temperature=self.temperature)
        return list(set(outputs))


def construct_chitchat_context(last_utterance_interpretation, last_utterance_labels, messages, max_depth=10):
   print(last_utterance_labels, 'факт внутри чит-чат контекста')
   labels2 = []
   if last_utterance_labels:
       for x in last_utterance_labels:
           if x[-1] not in '.?!':
               labels2.append(x+'.')
           else:
               labels2.append(x)

   if labels2:
       last_utterance_labels_txt = '[{}]'.format(' '.join(labels2))
   else:
       last_utterance_labels_txt = None
   steps = []
   for i, message in enumerate(messages):
       msg_text = ''.join(message)
       if i == len(messages)-1:
           if last_utterance_interpretation:
               msg_text = last_utterance_interpretation
           else:
               msg_text = msg_text

       prev_side = messages[i-1][0] if i > 0 else ''
       if prev_side != message[0]:
           steps.append(msg_text)
       else:
           s = steps[-1]
           if s[-1] not in '.?!;:':
               s += '.'
           steps[-1] = s + ' ' + msg_text

   print(last_utterance_labels_txt, 'last_utterance_labels_txt')
   if last_utterance_labels_txt:
       return steps[-max_depth:] + [last_utterance_labels_txt]
   else:
       return steps[-max_depth:]

def generate_pqa_reply(dialog, interpretation, processed_chitchat_contexts, premise_facts):
    chitchat_context = construct_chitchat_context(interpretation, premise_facts, dialog, max_depth=2)
    print(chitchat_context, 'контекст для чит-чата')
    chitchat_outputs = generative_model.generate_chitchat(context_replies=chitchat_context,
                                                                   num_return_sequences=1)

    return chitchat_outputs

generative_model = RugptChitchat()
generative_model.load('rugpt_chitchat')

if __name__ == '__main__':

    generative_model = RugptChitchat()
    generative_model.load('rugpt_chitchat')

    context = []
    dialog = ['Ю. Ты любишь ночь?','Б. я люблю долго спать','Ю. Ты любишь играть в игры?',
             'Б. мне нравятся компьютерные игры в жанре головоломок','Ю. Какой сон ты запомнила?','Б. я не помню свои сны']
    interpretation = 'Ты умеешь летать во сне?'
    premise_fact = ['Я не умею летать.']
    processed = set()
    print(generate_pqa_reply(dialog, interpretation, processed, premise_fact))
    # Интерактивная сессия - вводим реплики, модель генерирует ответные реплики, всё это накапливается в истории диалога.

    # while True:
    #     if context:
    #         print('Текущий диалог:'.format(len(context)))
    #         for i, s in enumerate(context, start=1):
    #             print('({})  {}'.format(i, s))
    #
    #     q = input(':> ').strip()
    #     print(q)
    #     if q:
    #         # Реплику с прикрепленной в конце предпосылкой разобьем на 2 части.
    #         m = re.match(r'^(.+) \[(.+)\]$', q)
    #         if m:
    #             text = m.group(1).strip()
    #             premise = m.group(2).strip()
    #             context.append(text)
    #             context.append('[' + premise + ']')
    #         else:
    #             context.append(q)
    #
    #         px = generative_model.generate_chitchat(context, num_return_sequences=5)
    #         print('Сгенерированные варианты ответа:')
    #         for i, p in enumerate(px):
    #             print('[{}]  {}'.format(i, p))
    #         print('')
    #         context.append(px[0])
    #     else:
    #         # Пустая реплика - значит надо начать новый диалог
    #         context = []


        # ENRICHED CHITCHAT
        # Интерактивная сессия с вводом вопроса и релевантной предпосылки для тестирования PQA-сценария.
    # while True:
    #     q = input('question:> ').strip()
    #     p = input('premise:>  ').strip()
    #     context = [q, '[' + p + '.]']
    #     px = generative_model.generate_chitchat(context, num_return_sequences=5)
    #     print('Сгенерированные варианты ответа:')
    #     for i, p in enumerate(px):
    #         print('[{}]  {}'.format(i, p))
    #     print('')

    # context = ['Привет, Вика!']
    # px = generative_model.generate_autoquestions(context, num_return_sequences=5)
    # print('Сгенерированные варианты автовопроса:')
    # for i, p in enumerate(px):
    #     print('[{}]  {}'.format(i, p))
    # print('')

    # context = ['{interpretation}', 'Какую музыку предпочитаешь?', 'Энергичную и мощную']
    # px = generative_model.generate_interpretations(context, num_return_sequences=5)
    # print('Сгенерированные варианты интерпретации:')
    # for i, p in enumerate(px):
    #     print('[{}]  {}'.format(i, p))
    # print('')


    # context = ['В какой стране живет Владимир Глуховский?']
    # px =generative_model.generate_confabulations(context, num_return_sequences=5)
    # print('Сгенерированные конфабуляции:')
    # for i, p in enumerate(px):
    #     print('[{}]  {}'.format(i, p))
    # print('')


    # while True:
    #     q = input('Question:> ').strip()
    #     px = generative_model.generate_confabulations([q], num_return_sequences=5)
    #     print('Сгенерированные конфабуляции:')
    #     for i, p in enumerate(px):
    #         print('[{}]  {}'.format(i, p))
    #     print('')


    # # НАЧАЛО ОТЛАДКИ
    # prompt_text = '<s>{chitchat}\n' + '- Как тебя зовут?\n[уклониться от ответа.]\n'
    # px = generative_model.generate_output_from_prompt(prompt_text, 5)
    # for i, p in enumerate(px):
    #     print('[{}]  {}'.format(i, p))
    # print('')
    # # КОНЕЦ ОТЛАДКИ
    #
    #
    #
    # context = ['Как ты относишься к украинцам?', '[уклониться от ответа.]']
    # px = generative_model.generate_chitchat(context, num_return_sequences=10)
    # print('Сгенерированные реплики диалога:')
    # for i, p in enumerate(px):
    #     print('[{}]  {}'.format(i, p))
    # print('')

