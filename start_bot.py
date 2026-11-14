import sys

if "--headless" in sys.argv[1:]:
    from deskbuddy.cli import main as headless_main

    raise SystemExit(headless_main())

from deskbuddy.settings import load_settings

local_only = load_settings().model_provider != "openai"

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


async def dispatch_reply(message, button_text, local_only, cortex=None):
    cortex = cortex or BotCortex
    if local_only:
        print("local only")
        if button_text == "Large":
            return await cortex.talk_history(message)
        elif button_text == "Small":
            return await cortex.talk_small(message)
        elif button_text == "Medium":
            return await cortex.talk_medium_with_history(message)
        else:
            return await cortex.talk_fast(message)
    else:
        print("not local only")
        if button_text == "Large":
            return await cortex.talk_history(message)
        elif button_text == "Small":
            return await cortex.talk_small(message)
        elif button_text == "Medium":
            return await cortex.talk_medium_API_with_history(message)
        else:
            return await cortex.talk_fast(message)


async def main():
    global local_only
    fast_button = tk.Button(root, text="Fast", command=lambda: chose_fast())
    small_button = tk.Button(root, text="Small", command=lambda: chose_small())
    medium_button = tk.Button(root, text="Medium", command=lambda: chose_medium())
    large_button = tk.Button(root, text="Large", command=lambda: chose_large())

    # Place the buttons in the main window
    fast_button.pack()
    small_button.pack()
    medium_button.pack()
    large_button.pack()

    local_only_tk = tk.BooleanVar()
    local_only_tk.set(local_only)
    local_only_checkbox = tk.Checkbutton(
        root, text="Local Only", variable=local_only_tk
    )
    local_only_checkbox.pack()

    # Start the main event loop
    root.mainloop()
    local_only = local_only_tk.get()
    await BotCortex.load_all_models(button_text, local_only)
    await helpers.load_models()

    poller = rc.TranscriptPoller()
    while True:
        await asyncio.sleep(0.1)
        try:
            message = await poller.next_transcript()
        except rc.ExecutionOutcomeUnknown as exc:
            print(f"Transcript delivery outcome is unknown ({exc.code}); continuing to poll.")
            continue
        except rc.TransportError as exc:
            print(f"Transcript request failed ({exc.code}); continuing to poll.")
            continue
        except (OSError, asyncio.TimeoutError) as exc:
            print(f"Transcript connection failed ({type(exc).__name__}); continuing to poll.")
            continue

        if message is not None:
            print("User: " + message)
            try:
                reply = await dispatch_reply(message, button_text, local_only)
            except Exception as exc:
                print(f"Could not generate a reply ({type(exc).__name__}); continuing to poll.")
            else:
                if not isinstance(reply, str):
                    print("Could not generate a text reply; continuing to poll.")
                    continue
                reply = reply.replace("\n", " ").strip()
                if reply:
                    print("Cooper: " + reply)
                    try:
                        await rc.send_reply(reply)
                    except rc.ExecutionOutcomeUnknown as exc:
                        print(
                            f"Ambiguous execution outcome for command '{exc.command}' "
                            f"(request_id={exc.request_id}): {exc}. "
                            "Will not automatically retry physical speech action."
                        )
                    except rc.TransportError as exc:
                        print(f"Could not confirm speech delivery ({exc.code}); continuing to poll.")
                    except Exception as exc:
                        print(
                            f"Could not confirm speech delivery ({type(exc).__name__}); "
                            "continuing to poll."
                        )


if __name__ == "__main__":
    asyncio.run(main())
