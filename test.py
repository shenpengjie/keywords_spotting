from wxpy import *
bot=Bot()
@bot.register(Group ,RECORDING)
def get_recording(msg):
    msg.get_file(''+msg.file_name)
    print(msg.file_name+'已下载')
embed()
