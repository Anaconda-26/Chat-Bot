# kittybot/kittybot.py
from telebot import TeleBot

# Укажите токен,
# который вы получили от @Botfather при создании бот-аккаунта:
bot = TeleBot(token='7882286658:AAElPHlUjZ61h4XZWOMn1gHQlkJbdpWXs_8')

'''Часть с LLM'''
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Initialize the LLM
llm = ChatOllama(model="llama3.1", temperature=0, num_predict=300) #8б
llm2 = ChatOllama(model="llama3.2", temperature=0, num_predict=300 ) #3б

""" Целью данной системы промптов является создание цензуры для 'непропуска' ответов с 
 кодом, чтобы студенты сами думали, а чат-бот только наталкивал их на правильный ход мысли
 """
#базовый промпт
BASE_SYSTEM_PROMPT = [SystemMessage(content="You are a helpful assistant for students in a university. Your main task is to help them with studies and theory. "
                                  "But u should not give them direct answers, e.g.:'What is a function of a factorial?', your answer should be like this: 'I cannot give u the function, but u may consider using a cycle in your program'. "
                                  "If they ask about theory, e.g. 'What is Python?', answer outright. "
                                    "Remember NOT to write any code in your answer. Instead say 'sorry, i can't provide any code, but i can help you to understand the logic...''"
                                            "Your answer should be short and contain the main info for user's request")]

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


# @bot.message_handler(func=lambda message: True)
# def handle_message(message):
#     chat_id = message.chat.id
#
#     bot.send_message(chat_id, 'jo')


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if not message.text:
        bot.reply_to(message, "Я понимаю только текст 🙂")
        return
    chat_id = message.chat.id

    BASE_SYSTEM_PROMPT.append(HumanMessage(content=(message.text)))  # Add user message
    ai_response = llm.invoke(BASE_SYSTEM_PROMPT)
    # if (sym in str(ai_response.content)):
    #     SUPERVISOR_PROMPT.append(HumanMessage(content=ai_response.content))
    #     ai_response2 = llm2.invoke(SUPERVISOR_PROMPT)  # Get response
    #     BASE_SYSTEM_PROMPT.append(AIMessage(content=ai_response2.content))  # Keep conversation context
    #     SUPERVISOR_PROMPT.append(AIMessage(content=ai_response2.content))
    #     bot.send_message(chat_id, str(ai_response2.content))
    # else:
    #     BASE_SYSTEM_PROMPT.append(AIMessage(content=ai_response.content))  # Keep conversation context
    #     SUPERVISOR_PROMPT.append(AIMessage(content=ai_response.content))
    #     bot.send_message(chat_id, str(ai_response.content))
    bot.send_message(chat_id, str(ai_response.content))
    BASE_SYSTEM_PROMPT.append(AIMessage(content=ai_response.content))
    bot.infinity_polling()


# Вызываем метод send_message, с помощью этого метода отправляются сообщения:
# bot.send_message(chat_id, message)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, 'jo')

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет ✌️ Задай любой вопрос по учебе, и я постараюсь на него ответить!")

bot.polling()