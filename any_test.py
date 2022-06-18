import rutokenizer
clause_splitter = rutokenizer.Segmenter()

def split_message_text(message):
    assertions = []
    questions = []
    for clause in split_clauses(message):
        if clause.endswith('?'):
            questions.append(clause)
        else:
            assertions.append(clause)
    return assertions, questions

def split_clauses (s):
    return list(clause_splitter.split(s))


text = 'Здорово ! Меня зовут Илья . А тебя как ?'
assertionx, questionx = split_message_text(text)

input_clauses = [(q, 1.0, True) for q in questionx] + [(a, 0.8, False) for a in assertionx]
print(input_clauses)
print(assertionx)
print(questionx)