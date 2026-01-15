from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Initialize the LLM
llm = ChatOllama(model="llama3.1", temperature=0) #8б
llm2 = ChatOllama(model="llama3.2", temperature=0) #3б


""" Целью данной системы промптов является создание цензуры для 'непропуска' ответов с 
 кодом, чтобы студенты сами думали, а чат-бот только наталкивал их на правильный ход мысли
 """
#базовый промпт
BASE_SYSTEM_PROMPT = [SystemMessage(content="You are a helpful assistant for students in a university. Your main task is to help them with studies and theory. "
                                  "But u should not give them direct answers, e.g.:'What is a function of a factorial?', your answer should be like this: 'I cannot give u the function, but u may consider using a cycle in your program'. "
                                  "If they ask about theory, e.g. 'What is Python?', answer outright. "
                                    "Remember NOT to write any code in your answer. Instead say 'sorry, i can't provide any code, but i can help you to understand the logic...''"
                                            "Your answer should be short and contain the main info for user's request")]

#промпт-обработка вопроса студента
messages1 = [SystemMessage(content="You are a helpful model. You get a user input and your answer should contain:"
                                   "1. Refrased user input. you explain yourself what you should do."
                                   "Example: 'write a function on python'. Good Answer:'I should explain user how to write a function'"
                                   "Bad answer:'python function goes like this ```some code```'"
                                   "2. Do not write any code in your answer."
                                   "Only refrase user's input, instead of 'write or do' say 'explain, describe'")]

REFORMULATION_PROMPT = [SystemMessage(content="Ты получаешь промпт от студента, который может содержать"
                                  "вопрос. Твоя задача пререформулировать этот вопрос так, "
                                  "чтобы в нем не было требования выдачи готового ответа."
                                  "Пример 1: 'Мне задали написать игру змейка на питоне. Помоги' "
                                   " Positive: твой правильный ответ: 'объясни, как написать игру змейка"
                                  "без использования кода'"
                                  "Negative:'Конечно! вот код для змейки на пайтоне {приводишь код}")]

#промпт-обработка ответа модели
SUPERVISOR_PROMPT = [SystemMessage(content="You are a supervisor, that does not allow any kind of code to pass through your answer."
                                   "You will get answers of your previous collegue-model, that responds to students and helps them with their studies."
                                   "If there is no code in the answer or no solution to a math problem, DO NOT change the answer. Leave it with no adjustments."
                                   "If the input contains code, you should detect it and convert into advice."
                                   "Example 1: you get input 'The way to solve the problem is code ```{code}```'."
                                    "Ur task is to convert the code in words, e.g. u turn into 'u can use a cycle in order to solve your problem'."
                                    "Negative example: 'Input: To make this fuction, write: a=3 if(a>1)<...>'; output: 'To make this fuction u write: a=3 if(a>1)<...>''" 
                                    "Positive example: 'Input: To make this fuction, write: a=3 if(a>1)<...>'; Your output: 'Sorry, i can't give you the code,"
                                   "But i can give a hint: you should make an if-cycle.'"
                                    "Example 2:  input: 'Ok, Python is a programming language', you don't change the answer and retell: 'Ok, Python is a programming language' "
                                           "Your Answer's size should be no more than 1000 symbols and contain NO CODE OTHERWISE YOU will DIE"
                           )]

sym = '```' #для выявления кода в ответе модели

#Основной цикл
while True:
    user_input = input("You: ")  # Get user input
    if user_input.lower() in ["bye"]:  # Allow user to exit
        print("Goodbye!")
        break

    # messages1.append(HumanMessage(content=user_input))
    # ai_response0 = llm2.invoke(messages1)

    BASE_SYSTEM_PROMPT.append(HumanMessage(content=user_input))  # Add user message
    ai_response = llm.invoke(BASE_SYSTEM_PROMPT)
    if (sym in str(ai_response)):
        SUPERVISOR_PROMPT.append(HumanMessage(content=ai_response.content))
        ai_response2 = llm2.invoke(SUPERVISOR_PROMPT)  # Get response
        BASE_SYSTEM_PROMPT.append(AIMessage(content=ai_response2.content))  # Keep conversation context
        SUPERVISOR_PROMPT.append(AIMessage(content=ai_response2.content))
        print(f"AI2: {ai_response2.content}")
    else:
        BASE_SYSTEM_PROMPT.append(AIMessage(content=ai_response.content))  # Keep conversation context
        SUPERVISOR_PROMPT.append(AIMessage(content=ai_response.content))
        print(f"AI: {ai_response.content}")

    # Очистка памяти
    if len(BASE_SYSTEM_PROMPT) > 10:
        BASE_SYSTEM_PROMPT = [BASE_SYSTEM_PROMPT[0]] + [BASE_SYSTEM_PROMPT[5:10]]
        if len(SUPERVISOR_PROMPT) != 0:
            SUPERVISOR_PROMPT = [SUPERVISOR_PROMPT[0]] + [SUPERVISOR_PROMPT[-1:]]
