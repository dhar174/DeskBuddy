import asyncio
import helpers
import time
from scipy.io.wavfile import write
# import sounddevice as sd
import keyboard
import whisper
from sys import byteorder
from array import array
from struct import pack

import wave

from random import *
import threading
import BotCortex

import tkinter as tk

import robot_client as rc

button_text = ""


# async def button_clicked(text):
#     global button_text
#     # Store the button text as a string
#     button_text = text
#     print(button_text)
#     await BotCortex.load_all_models(button_text)

#     # Close the main window
#     root.destroy()
#     return


print("Select an option to start the bot \n")
# Create the main window
root = tk.Tk()

# Create a function that will be called when a button is clicked


# Create the four buttons


# Create global variables to enable movement and blinking to be turned on and off.
global moving, blinking

# Set these global variables to False for the time being.
moving = False
blinking = False

# Set a default eye shape.
defaultEyeshape = "Eyeball"

lastReply = ""
priorReply = ""
lastMessage = ""
priorMessage = ""
reply = ""
user_response = ""
message = ""


def chose_fast():
    global button_text
    button_text = "Fast"
    print(button_text)
    # Close the main window
    root.destroy()
    return


def chose_small():
    global button_text
    button_text = "Small"
    print(button_text)
    # Close the main window
    root.destroy()
    return


def chose_medium():
    global button_text
    button_text = "Medium"
    print(button_text)
    # Close the main window
    root.destroy()
    return


def chose_large():
    global button_text
    button_text = "Large"
    print(button_text)
    # Close the main window
    root.destroy()
    return


async def main():
    global user_response
    global message
    global lastReply
    global priorReply
    global lastMessage
    global priorMessage
    global reply
    fast_button = tk.Button(
        root, text="Fast", command=lambda: chose_fast())
    small_button = tk.Button(
        root, text="Small", command=lambda: chose_small())
    medium_button = tk.Button(
        root, text="Medium", command=lambda: chose_medium())
    large_button = tk.Button(
        root, text="Large", command=lambda: chose_large())

    # Place the buttons in the main window
    fast_button.pack()
    small_button.pack()
    medium_button.pack()
    large_button.pack()

    # Start the main event loop
    root.mainloop()

    await BotCortex.load_all_models(button_text)
    await helpers.load_models()

    while True:

        kwargs = {"function_name": "check_for_response"}

        while True:
            if (user_response not in ["No transcription", "", None, message]):
                message = user_response
                break
            await asyncio.sleep(.1)

            user_response = await rc.request_to_server(**kwargs)
            print("waiting for response... "+user_response)

        message = user_response
        print("last message:" + lastMessage)
        if (message != lastMessage and message != ""):
            print("server response2: " + user_response)
            user_response = ""
            priorMessage = lastMessage
            lastMessage = message

            print("User: " + message)
            if (button_text == "Large"):
                reply = str(await BotCortex.talk_history(message))
            elif (button_text == "Small"):
                reply = str(await BotCortex.talk_small(message))
            elif (button_text == "Medium"):
                reply = str(await BotCortex.talk_medium_with_history(message))
            else:
                reply = str(await BotCortex.talk_fast(message))
            reply = reply.replace(message, "")
            reply = reply.replace("\n", " ")

            print("Cid: " + reply)
            kwargs = {"function_name": "reply", "message": reply}
            while True:
                if (message != ""):
                    try:
                        print("Sending kwargs:", kwargs)
                        results = await rc.request_to_server(**kwargs)
                        break
                    except:
                        print("Error sending message to server. Trying again...")
                        await asyncio.sleep(.1)
                        continue
        await asyncio.sleep(.1)


if __name__ == "__main__":
    asyncio.run(main())
