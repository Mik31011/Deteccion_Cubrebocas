import requests
import logging
logging.basicConfig(level= logging.DEBUG, format= '[%(levelname)s] (%(threadName)-s), %(message)s')
def Message(destinations, message):
    try:
        logging.info("Intentando")
        payload = [
            ('cmd', 'sendsms'), 
            ('domainId', 'XX'),
            ('login', 'miguelsdlg2112@gmail.com'),
            ('passwd', '7fcangxb'),
            ('msg', message),
            ('senderId', 'Miguel'),
            ('dest', destinations)
        ]
        contentType = {'Content-Type':'application/x-www-form-urlencoded;charset=utf-8'}
        url = 'http://www.altiria.net/api/http'
        r = requests.post(url, data= payload, headers= contentType, timeout= (5,60))
        print("Mensaje enviado")
        return r.text
    except requests.ConnectTimeout:
        print("Tiempo de conexión con el servidor agotado")
    except requests.ReadTimeout:
        print("Tiempo de respuesta del servidor agotado")
    except Exception as ex:
        print("Error interno: " + str(ex) )

destinatario = "522462115937"
Mensaje = "Que pedo pinche zangano"
print("Intentando enviar mensaje")
Response = Message(destinations= destinatario, message= Mensaje)
print(Response)
			
