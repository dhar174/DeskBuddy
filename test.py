import BlenderBot1b


def main():
    BlenderBot1b.init()
    while True:
        print("Enter a message to send to BlenderBot1b:")
        message = input()
        print("BlenderBot1b: " + BlenderBot1b.talk_history(message))


if __name__ == "__main__":
    main()
