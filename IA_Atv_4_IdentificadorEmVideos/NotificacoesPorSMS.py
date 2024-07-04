# Código para enviar um SMS a partir do Twillio
# https://www.twilio.com/pt-br | https://www.twilio.com/pt-br/sms/pricing/us
# Você pode utilizar outra ferramenta para enviar SMS
# Importante se trata, da notificação a você chegar
from twilio.rest import Client

# Twilio, detalhes da sua conta, não compartilhe este arquivo de preferência
twilio_account_sid = 'Seu SID do Twilio aqui'
twilio_auth_token = 'Seu Token de Autenticação Twilio aqui'
twilio_source_phone_number = 'Seu número de telefone do Twilio aqui'

# Cria um objeto cliente do Twilio para uma instância
client = Client(twilio_account_sid, twilio_auth_token)

# Envia um SMS
message = client.messages.create(
    body="Essa é a mensagem da vaga disponível!",
    from_=twilio_source_phone_number,
    to="Número de telefone de destino aqui"
)
