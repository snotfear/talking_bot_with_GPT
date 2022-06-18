import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


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
            total_sequence = text[text.find('#') + 1:].strip()
            # total_sequence = text[len(prompt_text)-1:].strip()  # отрезаем промт
            if total_sequence:
                generated_sequences.add(total_sequence)
        return list(generated_sequences)


def generate_interpretations(model, context_replies, num_return_sequences):
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
    outputs = model.generate_output_from_prompt(prompt_text, num_return_sequences)
    return list(set(outputs))


def constuct_interpreter_contexts(messages):
    contexts = set()
    max_history = 2
    messages2 = [m for m in messages]
    for n in range(2, max_history+2): # несколько кругов для составления разных вариантов генерации интерпритаций
        steps = []
        for i, message in enumerate(messages2):
            msg_text = message[2:] # Отсекаем букву с точкой говорящего (Ю. - юзер, Б. - бот)
            prev_side = messages2[i-1][0] if i > 0 else ''
            if prev_side != message[0]: # Проверка на последовательность реплик
                steps.append(msg_text) # Если реплика следующая реплика от второго участника диалога, то добавляем её в список
            else:
                s = steps[-1]
                if s[-1] not in '.?!;:': # А если две реплики идут подряд от одного пользователя (или бота), то сращиваем их в одну
                    s += '.'
                steps[-1] = s + ' ' + msg_text
        last_steps = steps[-n:]
        context = ' | '.join(last_steps) # Разделяем стороны диалога чертой.
        contexts.add(context)
    return sorted(list(contexts), key=lambda s: -len(s))





if __name__ == '__main__':
    user_q = generate_interpretations(context_replies=['Меня зовут Лита. | красивое имя'], num_return_sequences=5)
    print(user_q)