from concurrent.futures import ThreadPoolExecutor
from sys import byteorder
import asyncio
import pickle
import signal
import sys
import threading
import wave
from array import array
from random import randint, random
from struct import pack

import aiohttp
import pyaudio
import keyboard

from picoh import picoh
import whisper

connector = aiohttp.TCPConnector(force_close=True)
# Define a TCP server
HOST = "0.0.0.0"
PORT = 8888


class MyFunctions:
    _public_methods_ = [
        "simple_test",
        "button_clicked",
        "send_response",
        "listen",
        "baseCol",
        "randomNod",
        "randomTurn",
        "randomLook",
        "blinkLids",
        "record_to_file",
        "record",
        "add_silence",
        "trim",
        "normalize",
        "is_silent",
        "button_clicked",
    ]

    async def simple_test():
        print("Hello")
        return "Hello"

    async def check_for_response():
        global lastTranscription
        global priorTranscription
        # print("Checking for response")

        if lastTranscription != "" and lastTranscription != priorTranscription:
            print(priorTranscription)
            print(lastTranscription)
            return lastTranscription
        else:
            return "No transcription"

    async def blinkLids():
        global blinking

        while True:
            # for i in range(10):
            #     await asyncio.sleep(1)
            if blinking:
                for x in range(10, 0, -1):
                    await picoh.move(picoh.LIDBLINK, x)
                    await picoh.wait(0.01)

                for x in range(0, 10):
                    await picoh.move(picoh.LIDBLINK, x)
                    await picoh.wait(0.01)

                await picoh.wait(random() * 6)
            # await asyncio.sleep(1)

    async def randomLook():
        global moving
        while True:
            # for i in range(10):
            #     await asyncio.sleep(1)
            if moving:
                await picoh.move(picoh.EYETILT, randint(2, 8))
                await picoh.move(picoh.EYETURN, randint(2, 8))

                await picoh.wait(random() * 5)
            # await asyncio.sleep(1)

    async def randomTurn():
        global moving
        while True:
            # for i in range(10):
            # await asyncio.sleep(1)
            if moving:
                await picoh.move(picoh.HEADTURN, randint(3, 7))

                await picoh.wait(random() * 4)
            # await asyncio.sleep(1)

    async def randomNod():
        global moving
        while True:
            # for i in range(10):
            #     await asyncio.sleep(1)
            if moving:
                await picoh.move(picoh.HEADNOD, randint(4, 7))

                await picoh.wait(random() * 4)
            # await asyncio.sleep(1)

    async def baseCol(r, g, b):
        await picoh.setBaseColour(r, g, b)
        return

    async def picoh_say(text):
        print("in picoh_say")
        print(str(text))
        await picoh.say(str(text))
        return

    async def picoh_move(motor, position):
        await picoh.move(motor, position)
        return

    async def picoh_wait(seconds):
        await picoh.wait(seconds)
        return

    async def picoh_get_input_path():
        return await picoh.get_input_path()

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

    async def check_key_pressed(key, delay=0.1):
        while True:
            if keyboard.is_pressed(key):
                return True
            await asyncio.sleep(delay)

    @staticmethod
    async def listen():
        global listening
        global recording
        global recorded
        global result
        global lastReplyMessage
        recorded = False
        recording = False
        result = None
        waiting_for_reply = False
        while True:
            try:
                if listening:
                    fs = 44100

                    if (
                        not waiting_for_reply
                        and not recording
                        and await MyFunctions.check_key_pressed("space")
                    ):
                        await MyFunctions.baseCol(0, 8, 0)
                        print("Recording started")
                        if not recorded and await MyFunctions.check_key_pressed(
                            "space"
                        ):
                            input_path = await picoh.get_input_path()
                            sample_width, data = MyFunctions.record()
                            t = threading.Thread(
                                target=MyFunctions.record_to_file_sync,
                                args=(input_path, sample_width, data),
                            )
                            t.daemon = True
                            t.start()
                            waiting_for_reply = True

                if not recording and recorded:
                    print("Transcribing...")
                    result = model.transcribe(await MyFunctions.picoh_get_input_path())
                    await MyFunctions.picoh_wait(0.25)
                    print(result["text"])
                    waiting_for_reply = False

                if result and result["text"].replace(" ", "") != "":
                    resp = await MyFunctions.send_response(result["text"])
                    if resp:
                        waiting_for_reply = False
                else:
                    recorded = False
                    recording = False
                    result = None
                    await MyFunctions.baseCol(0, 0, 5)
            except Exception as e:
                print(e)
                pass
            await asyncio.sleep(0.1)  # Add a small sleep to prevent high CPU usage
        return None

    def is_silent(snd_data):
        # "Returns 'True' if below the 'silent' threshold"
        return max(snd_data) < THRESHOLD

    def normalize(snd_data):
        "Average the volume out"
        MAXIMUM = 16384
        times = float(MAXIMUM) / max(abs(i) for i in snd_data)

        r = array("h")
        for i in snd_data:
            r.append(int(i * times))
        return r

    def trim(snd_data):
        # "Trim the blank spots at the start and end"
        def _trim(snd_data):
            snd_started = False
            r = array("h")

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
        r = array("h", silence)
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
        stream = p.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        num_silent = 0
        snd_started = False

        r = array("h")

        while 1:
            # little endian, signed short
            snd_data = array("h", stream.read(CHUNK_SIZE))
            if byteorder == "big":
                snd_data.byteswap()
            r.extend(snd_data)
            silent = MyFunctions.is_silent(snd_data)
            if silent and snd_started:
                num_silent += 1
            elif not silent and not snd_started:
                snd_started = True
            if snd_started and num_silent > 80:
                break

        sample_width = p.get_sample_size(FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()

        r = MyFunctions.normalize(r)
        r = MyFunctions.trim(r)
        r = MyFunctions.add_silence(r, 0.5)
        return sample_width, r

    async def record_to_file(path, sample_width, data):
        "Records from the microphone and outputs the resulting data to 'path'"
        global recording
        global recorded
        recording = True
        # try:
        #     # sample_width, data = MyFunctions.record()
        #     # t = threading.Thread(target=MyFunctions.record_to_file_sync,
        #     #                      args=(path, sample_width, data))
        #     # t.start()
        #     data = pack('<' + ('h'*len(data)), *data)

        #     wf = wave.open(path, 'wb')
        #     wf.setnchannels(1)
        #     wf.setsampwidth(sample_width)
        #     wf.setframerate(RATE)
        #     wf.writeframes(data)
        #     wf.close()
        # except Exception as e:
        #     print("Recording failed:", e)
        #     recording = False
        #     recorded = False
        #     return
        recorded = True
        recording = False
        await MyFunctions.baseCol(0, 0, 5)
        print("Recording stopped")
        return None

    def record_to_file_sync(path, sample_width, data):
        global recording
        global recorded
        try:
            data = pack("<" + ("h" * len(data)), *data)
            wf = wave.open(path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(RATE)
            wf.writeframes(data)
            wf.close()
        except Exception as e:
            print("Error saving the file:", e)
            recording = False
            recorded = False
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            MyFunctions.record_to_file(path, sample_width, data)
        )

    def get_input_path_sync():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(picoh.get_input_path())

    def record_sync(duration):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(MyFunctions.record(duration))


async def start_server():
    try:
        server = await asyncio.start_server(handle_client, HOST, PORT)
    except Exception as e:
        print(f"Error starting server: {e}")
        return
    async with server:
        print(f"Server started on {HOST}:{PORT}")
        await server.serve_forever()


# Set the moving and blinking global variables to True.


moving = True
blinking = True
listening = True

# Set these global variables to False for the time being.
moving = False
blinking = False

# Set a default eye shape.
defaultEyeshape = "Eyeball"

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100


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
            function_name = payload["function_name"]
            # print(f"Received function call: {function_name}")

            if "message" in payload:
                message = payload["message"]
                func = getattr(MyFunctions, function_name)
                results = await func(message)
            else:
                func = getattr(MyFunctions, function_name)
                results = await func()

            writer.write(pickle.dumps(results))
            await writer.drain()
            for i in range(10):
                if i == 9:
                    await asyncio.sleep(0.1)

        writer.close()
        await writer.wait_closed()
        return None
    except Exception as e:
        print(f"Error handling client: {e}")
        return None


def signal_handler(signal, frame):
    loop.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


async def main():
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
    global firstLoop
    global blinking
    global moving
    blinking = True
    moving = True
    if firstLoop:
        await picoh.reset()
        await picoh.wait(1)
        await picoh.setEyeBrightness(1)
        await picoh.setEyeShape(defaultEyeshape)

    await picoh.say(
        "Hello my name is Cooper. Cooper stands for Conversational Office Organizer and Personal Entertainment Robot. Please give me a few moments to start up."
    )
    print("Starting up...")
    server_task = asyncio.ensure_future(start_server())
    print("Server starting")
    listen_task = asyncio.ensure_future(MyFunctions.listen())
    blink_task = asyncio.ensure_future(MyFunctions.blinkLids())
    look_task = asyncio.ensure_future(MyFunctions.randomLook())
    nod_task = asyncio.ensure_future(MyFunctions.randomNod())
    turn_task = asyncio.ensure_future(MyFunctions.randomTurn())
    print("Tasks starting")
    await asyncio.gather(
        server_task, listen_task, blink_task, look_task, nod_task, turn_task
    )

    print("Tasks started")
    print("Tasks finished")


import concurrent.futures

from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":
    firstLoop = True

    model = whisper.load_model("base")
    # loop = asyncio.get_event_loop()
    lastReplyMessage = ""
    priorReplyMessage = ""
    lastTranscription = ""
    priorTranscription = ""
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    RATE = 44100
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        loop.set_default_executor(executor)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()
    # asyncio.run(main())
    # loop.run_forever()
