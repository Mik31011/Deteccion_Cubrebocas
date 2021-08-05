import time
import schedule
import requests

bot_token = '1634706110:AAENkGoKMX0iXyJ8YNyglUajcfrs7fSUoOA'
bot_chatID = '1094906652'

def telegram_bot_sendtxt(bot_message):
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return response.json()


Mensaje = "El ciclo de entrenamiento ha concluido perro"

End_While = 3
while End_While <= 3:
    End_While += 1
    print(telegram_bot_sendtxt(Mensaje))
    print("Enviando alerta")
    time.sleep(2)
