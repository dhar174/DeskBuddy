
import asyncio
import pickle
import socket
import tkinter as tk
import threading
from random import *
import time
from scipy.io.wavfile import write
# import sounddevice as sd
import keyboard
import whisper
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

from picoh import picoh
# import BlenderBot1b
# import pythonwin
import signal
import sys
import aiohttp

connector = aiohttp.TCPConnector(force_close=True)


class MyFunctions:
    _public_methods_ = ['simple_test', 'button_clicked', "send_response", "listen", "baseCol", "randomNod",
                        "randomTurn", "randomLook", "blinkLids", "record_to_file", "record", "add_silence", "trim", "normalize", "is_silent", "button_clicked"]  # COM exposed methods

    async def simple_test():
        print("Hello")
        return "Hello"

    async def check_for_response():
        global lastTranscription
        global priorTranscription
        print("Checking for response")

        if lastTranscription != "" and lastTranscription != priorTranscription:
            print(priorTranscription)
            print(lastTranscription)
            return lastTranscription
        else:

            return "No transcription"

    async def blinkLids():
        global blinking

        # While True - Loop forever.
        while True:
            for i in range(10):
                await asyncio.sleep(1)
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
            await asyncio.sleep(1)

    async def randomLook():
        global moving
        while True:
            for i in range(10):
                await asyncio.sleep(1)
            # if moving is True.
            if moving:
                # Look in a random direction.
                picoh.move(picoh.EYETILT, randint(2, 8))
                picoh.move(picoh.EYETURN, randint(2, 8))

                # Wait for between 0 and 5 seconds.
                picoh.wait(random() * 5)
            await asyncio.sleep(1)

    async def randomTurn():
        global moving
        while True:
            for i in range(10):
                await asyncio.sleep(1)
            if moving:
                # Move Picoh's HEADTURN motor to a random position between 3 and 7.
                picoh.move(picoh.HEADTURN, randint(3, 7))

                # wait for a random amount of time before moving again.
                picoh.wait(random() * 4)
            await asyncio.sleep(1)

    async def randomNod():
        global moving
        while True:
            for i in range(10):
                await asyncio.sleep(1)
            if moving:

                # Move Picoh's HEADNOD motor to a random position between 4 and 7.
                picoh.move(picoh.HEADNOD, randint(4, 7))

                # wait for a random amount of time before moving again.
                picoh.wait(random() * 4)
            await asyncio.sleep(1)

    async def baseCol(r, g, b):

        # Set the base to a random rgb values between 0 and 10.
        picoh.setBaseColour(r, g, b)
        return

        # Wait between 10 and 20 seconds before changing again.
    async def picoh_say(text):
        print("in picoh_say")
        print(str(text))
        picoh.say(str(text))
        return

    async def picoh_move(motor, position):
        picoh.move(motor, position)
        return

    async def picoh_wait(seconds):
        picoh.wait(seconds)
        return

    async def picoh_get_input_path():
        return picoh.get_input_path()
    THRESHOLD = 500

    async def send_response(text):

        global recording
        global recorded
        global result
        global lastTranscription
        global priorTranscription
        global lastReplyMessage
        global priorReplyMessage
        print("in send_response")
        recorded = False
        recording = False
        priorTranscription = lastTranscription
        lastTranscription = text
        result = None
        return

    async def reply(text):
        global lastReplyMessage
        global priorReplyMessage
        print("Replying: " + text)
        if text != lastReplyMessage and text != "" and text != None:
            print(text)
            priorReplyMessage = lastReplyMessage
            lastReplyMessage = text
            await MyFunctions.baseCol(8, 0, 0)
            await MyFunctions.picoh_wait(0.1)
            print("about to speak")
            await MyFunctions.picoh_say(text)
            await MyFunctions.picoh_wait(0.1)
        return

    async def is_silent(snd_data):
        # "Returns 'True' if below the 'silent' threshold"
        return max(snd_data) < THRESHOLD

    async def normalize(snd_data):
        "Average the volume out"
        MAXIMUM = 16384
        times = float(MAXIMUM)/max(abs(i) for i in snd_data)

        r = array('h')
        for i in snd_data:
            r.append(int(i*times))
        return r

    async def trim(snd_data):
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

    async def add_silence(snd_data, seconds):
        "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
        silence = [0] * int(seconds * RATE)
        r = array('h', silence)
        r.extend(snd_data)
        r.extend(silence)
        return r

    async def record():
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
            silent = await MyFunctions.is_silent(snd_data)
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

        r = await MyFunctions.normalize(r)
        r = await MyFunctions.trim(r)
        r = await MyFunctions.add_silence(r, 0.5)
        return sample_width, r

    async def record_to_file(path):
        "Records from the microphone and outputs the resulting data to 'path'"
        global recording
        global recorded
        recording = True
        try:
            sample_width, data = await MyFunctions.record()
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
        await MyFunctions.baseCol(0, 0, 5)
        print("Recording stopped")
        return None

    async def listen():
        global listening
        global recording
        global recorded
        global result
        global lastReplyMessage
        recorded = False
        recording = False
        result = None
        while True:
            try:
                if listening:
                    fs = 44100
                    if keyboard.is_pressed("space") and recording == False:
                        await MyFunctions.baseCol(0, 8, 0)
                        print("Recording started")
                        if keyboard.is_pressed("space") and recorded == False:
                            await MyFunctions.record_to_file(picoh.get_input_path())
                # else:
                #     time.sleep(0.1)
                if (recording != True and recorded == True):
                    print("Transcribing...")
                    # wav = wave.open(picoh.get_input_path(), 'rb')
                    result = model.transcribe(
                        await MyFunctions.picoh_get_input_path())
                    await MyFunctions.picoh_wait(0.25)
                    print(result["text"])
                if (result != None and result["text"].replace(" ", "") != ""):
                    resp = await MyFunctions.send_response(
                        result["text"])
                else:
                    recorded = False
                    recording = False
                    result = None
                    await MyFunctions.baseCol(0, 0, 5)
            except Exception as e:
                print(e)
                pass
            for i in range(10):
                if (i == 9):
                    await asyncio.sleep(0.1)
        return None  # This is required to stop the loop


# Define a TCP server
HOST = '0.0.0.0'
PORT = 8888


async def handle_client(reader, writer):
    global lastReplyMessage
    global priorReplyMessage
    results = ""
    message = ""
    try:
        while True:
            data = await reader.read(1024)
            if not data:
                break

            payload = pickle.loads(data)
            function_name = payload['function_name']
            # print(f"Received function call: {function_name}")

            if 'message' in payload:
                message = payload['message']
                func = getattr(MyFunctions, function_name)
                results = await func(message)
            else:
                func = getattr(MyFunctions, function_name)
                results = await func()

            writer.write(pickle.dumps(results))
            await writer.drain()
            for i in range(10):
                if (i == 9):
                    await asyncio.sleep(0.1)

        writer.close()
        await writer.wait_closed()
        return None
    except Exception as e:
        print(f"Error handling client: {e}")
        return None


async def start_server():
    try:
        server = await asyncio.start_server(handle_client, HOST, PORT)
    except Exception as e:
        print(f"Error starting server: {e}")
        return
    async with server:
        print(f"Server started on {HOST}:{PORT}")
        await server.serve_forever()
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #     s.bind((HOST, PORT))
    #     s.listen()
    #     print(f"Server started on {HOST}:{PORT}")
    #     while True:
    #         # Wait for a new connection

    #         try:
    #             conn, addr = s.accept()

    #             # print(f"1New connection from {addr[0]}:{addr[1]}")
    #             # Spawn a new thread to handle the connection
    #             # t = threading.Thread(target=handle_client, args=(conn, addr))
    #             # t.start()
    #             # await handle_client(conn, addr)
    #             loop.create_task(handle_client(conn, addr))
    #         except KeyboardInterrupt:
    #             print("Server stopped")
    #             break
    #         await asyncio.sleep(0.1)


# Create the main window

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


# Reset Picoh and wait for a second for motors to move to reset positions.
picoh.reset()
picoh.wait(1)
picoh.setEyeShape(defaultEyeshape)

# Set the moving and blinking global variables to True.


moving = True
blinking = True
listening = True

# servertask = threading.Thread(target=start_server, daemon=True)
# t0 = threading.Thread(target=MyFunctions.listen,
#                       daemon=True, name='Listen')


# # t0 = threading.Thread(target=MyFunctions.listen, args=())

# # Create a thread for blinking.
# t1 = threading.Thread(target=MyFunctions.blinkLids)

# # Create a thread to make eyes look around randomly.
# t2 = threading.Thread(target=MyFunctions.randomLook)

# # Create a thread for random head nod positions.
# t3 = threading.Thread(target=MyFunctions.randomNod)

# # Create a thread for random head turn positions.
# t4 = threading.Thread(target=MyFunctions.randomTurn)

# Create a thread for random base colour.
# t5 = threading.Thread(target=baseCol, args=())
picoh.say(
    "Hello my name is Cooper. Please give me a few moments to start up.")

# Start the threads.

# t5.start()


def signal_handler(signal, frame):
    loop.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


async def main():
    global servertask
    global t0
    global t1
    global t2
    global t3
    global t4
    global t5
    global picoh
    global lastReplyMessage
    global priorReplyMessage
    global lastTranscription
    global priorTranscription
    global CHUNK_SIZE
    global FORMAT
    global RATE
    global listening
    global model
    global moving
    global blinking
    global defaultEyeshape
    global THRESHOLD
    global recording
    global recorded

    # asyncio.ensure_future(start_server())
    # loop.create_task(start_server())
    # asyncio.ensure_future(MyFunctions.listen())
    # asyncio.ensure_future(MyFunctions.blinkLids())
    # asyncio.ensure_future(MyFunctions.randomLook())
    # asyncio.ensure_future(MyFunctions.randomNod())
    # asyncio.ensure_future(MyFunctions.randomTurn())
    # loop.run_forever()
    server_task = asyncio.create_task(start_server())
    listen_task = asyncio.create_task(MyFunctions.listen())
    blink_task = asyncio.create_task(MyFunctions.blinkLids())
    look_task = asyncio.create_task(MyFunctions.randomLook())
    nod_task = asyncio.create_task(MyFunctions.randomNod())
    turn_task = asyncio.create_task(MyFunctions.randomTurn())

    await asyncio.gather(server_task, listen_task)
    # t0.start()
    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()


# Start the main event loop.

if __name__ == "__main__":
    model = whisper.load_model("base")
    loop = asyncio.get_event_loop()

    lastReplyMessage = ""
    priorReplyMessage = ""
    lastTranscription = ""
    priorTranscription = ""
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    RATE = 44100

    # picoh.say(
    #     "Hi there! How are you doing today? Don't answer that, I still need a few more seconds to start up.")
    # picoh.say(
    #     "few more seconds to start up.")
    # # servertask.start()
    # # t5 = threading.Thread(target=start_server, args=())
    # # t5.start()
    # picoh.say("Okee dokee. I am ready to talk!")
    asyncio.run(main())
    loop.run_forever()
