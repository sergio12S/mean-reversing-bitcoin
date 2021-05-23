from dotenv import dotenv_values

from telethon import TelegramClient, events
# from telethon.tl.functions.messages import (GetHistoryRequest)


config = dotenv_values(".env")
user_input_channel = 'https://t.me/whale_alert_io'

client = TelegramClient(
    'anon',
    config.get('API_TELEGA'),
    config.get('API_HASH_TELEGA')
)


@client.on(events.NewMessage(chats=user_input_channel))
async def messagerListener(event):
    message = event.message.message
    print(message)
    await client.forward_messages(entity='me', messages=event.message)

with client:
    client.run_until_disconnected()
