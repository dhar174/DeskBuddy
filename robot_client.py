import socket
import pickle
import asyncio

# Connect to the TCP server and send a request
HOST = '172.24.160.1'
PORT = 8888


async def request_to_server(**kwargs):
    # print("Sending request to server...")
    reader, writer = await asyncio.open_connection(HOST, PORT)

    try:

        writer.write(pickle.dumps(kwargs))
        await writer.drain()
        # print("Request sent, waiting for response...")
        data = await reader.read(1024)
        # print("Response received")
        results = pickle.loads(data)
        writer.close()
        await writer.wait_closed()
        # print("Connection closed")
        return results
    except asyncio.TimeoutError:
        print("Server not responding within 5 seconds")
        writer.close()
        await writer.wait_closed()
        return None
    except Exception as e:
        print(f"Error sending request to server: {e}")
        writer.close()
        await writer.wait_closed()
        return None

    # try:
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         s.connect((HOST, PORT))

    #         data = pickle.dumps((kwargs))
    #         s.sendall(data)
    #         result = pickle.loads(s.recv(1024))
    # except Exception as e:
    #     print(e)
    # return result


# kwargs = {"function_name": "simple_test"}

# r = request_to_server(**kwargs)
# print(r)
