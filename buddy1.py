# Example of threading using the picoh library.
# Threading allows Picoh to do multiple things at once.
# The program is split into functions, each of which runs on its own thread.
# This example will make Picoh look around randomly and blink at random intervals.

# Import the relevant libraries.
from scipy.io.wavfile import write
import sounddevice as sd
import keyboard
import whisper
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

from picoh import picoh
from random import *
import threading
import bot_management

import tkinter as tk

# Create the main window
root = tk.Tk()

# Create a function that will be called when a button is clicked

global button_text


def button_clicked(text):
    global button_text
    # Store the button text as a string
    button_text = text
    print(button_text)
    if (button_text == "Fast"):
        bot_management.init_fast()
    elif (button_text == "Small"):
        bot_management.init_small()
    elif (button_text == "Medium"):
        bot_management.init_medium()
    elif (button_text == "Large"):
        bot_management.init()

    # Close the main window
    root.destroy()


# Create the four buttons
fast_button = tk.Button(
    root, text="Fast", command=lambda: button_clicked("Fast"))
small_button = tk.Button(
    root, text="Small", command=lambda: button_clicked("Small"))
medium_button = tk.Button(
    root, text="Medium", command=lambda: button_clicked("Medium"))
large_button = tk.Button(
    root, text="Large", command=lambda: button_clicked("Large"))

# Place the buttons in the main window
fast_button.pack()
small_button.pack()
medium_button.pack()
large_button.pack()

# Start the main event loop
root.mainloop()

model = whisper.load_model("base")
# Create global variables to enable movement and blinking to be turned on and off.
global moving, blinking

# Set these global variables to False for the time being.
moving = False
blinking = False

# Set a default eye shape.
defaultEyeshape = "Eyeball"

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r


def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r


def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    global recording
    global recorded
    recording = True
    try:
        sample_width, data = record()
        data = pack('<' + ('h'*len(data)), *data)

        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()
    except:
        print("Recording failed")
        recording = False
        recorded = False
        return
    recorded = True
    recording = False
    baseCol(0, 0, 5)
    print("Recording stopped")


def blinkLids():
    global blinking

    # While True - Loop forever.
    while True:
        # if blinking is True.
        if blinking:
            # for the numbers 10 to 0 set the lidblink position.
            for x in range(10, 0, -1):
                picoh.move(picoh.LIDBLINK, x)
                picoh.wait(0.01)

            # for the numbers 0 to 10 set the lidblink position.
            for x in range(0, 10):
                picoh.move(picoh.LIDBLINK, x)
                picoh.wait(0.01)

            # wait for a random amount of time for realistic blinking
            picoh.wait(random() * 6)


def randomLook():
    global moving
    while True:
        # if moving is True.
        if moving:
            # Look in a random direction.
            picoh.move(picoh.EYETILT, randint(2, 8))
            picoh.move(picoh.EYETURN, randint(2, 8))

            # Wait for between 0 and 5 seconds.
            picoh.wait(random() * 5)


def randomTurn():
    global moving
    while True:
        if moving:
            # Move Picoh's HEADTURN motor to a random position between 3 and 7.
            picoh.move(picoh.HEADTURN, randint(3, 7))

            # wait for a random amount of time before moving again.
            picoh.wait(random() * 4)


def randomNod():
    global moving
    while True:
        if moving:

            # Move Picoh's HEADNOD motor to a random position between 4 and 7.
            picoh.move(picoh.HEADNOD, randint(4, 7))

            # wait for a random amount of time before moving again.
            picoh.wait(random() * 4)


def baseCol(r, g, b):

    # Set the base to a random rgb values between 0 and 10.
    picoh.setBaseColour(r, g, b)

    # Wait between 10 and 20 seconds before changing again.


# def record(fs):
#     global recording
#     global recorded
#     recorded = False
#     recording = True
#     # Set the recording parameters
#     # Sample rate

#     # Create an empty list to store the recorded audio
#     my_recording = []

#     # Start the recording
#     print("Press and hold the space key to start recording")

#     while True:
#         # Start recording when the space key is pressed

#         # Continue recording while the space key is pressed
#         while keyboard.is_pressed("space"):
#             my_recording.extend(
#                 sd.rec(int(fs / 10), samplerate=fs, channels=2))
#             print("Recording...")
#             sd.wait()  # Wait until recording is finished
#         print("Recording stopped")
#         recording = False
#         recorded = True
#         baseCol(0, 0, 5)
#         return my_recording

#     # Save the recorded audio as a WAV file


def listen():
    global listening
    global recording
    global recorded
    global result
    recorded = False
    recording = False
    result = None
    while True:
        if listening:
            fs = 44100
            if keyboard.is_pressed("space"):
                baseCol(0, 8, 0)
                print("Recording started")
                if keyboard.is_pressed("space") and recorded == False:
                    record_to_file(picoh.get_input_path())
        if (recording != True and recorded == True):
            print("Transcribing...")

            # wav = wave.open(picoh.get_input_path(), 'rb')
            result = model.transcribe(picoh.get_input_path())
            picoh.wait(0.25)
            print(result["text"])
        if (result != None and result["text"].replace(" ", "") != ""):
            respond(result["text"])
        else:
            recorded = False
            recording = False
            result = None
            baseCol(0, 0, 5)


def respond(text):
    global recording
    global recorded
    global result
    baseCol(8, 0, 0)
    if (button_text == "Large"):
        reply = bot_management.talk_history(text)
    elif (button_text == "Small"):
        reply = bot_management.talk_small(text)
    elif (button_text == "Medium"):
        reply = bot_management.talk_medium(text)
    else:
        reply = bot_management.talk_fast(text)
    picoh.wait(0.1)
    picoh.say(reply)
    picoh.wait(0.1)

    recorded = False
    recording = False
    result = None


# Reset Picoh and wait for a second for motors to move to reset positions.
picoh.reset()
picoh.wait(1)
picoh.setEyeShape(defaultEyeshape)

# Set the moving and blinking global variables to True.


moving = True
blinking = True
listening = True
t0 = threading.Thread(target=listen, args=())

# Create a thread for blinking.
t1 = threading.Thread(target=blinkLids, args=())

# Create a thread to make eyes look around randomly.
t2 = threading.Thread(target=randomLook, args=())

# Create a thread for random head nod positions.
t3 = threading.Thread(target=randomNod, args=())

# Create a thread for random head turn positions.
t4 = threading.Thread(target=randomTurn, args=())

# Create a thread for random base colour.
# t5 = threading.Thread(target=baseCol, args=())
picoh.say("Hello my name is Picoh. Please give me a few moments to start up.")

# Start the threads.
t0.start()
t1.start()
t2.start()
t3.start()
t4.start()
# t5.start()

picoh.say("Hi there! How are you doing today?")
