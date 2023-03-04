import BotCortex


def main():
    BotCortex.init()
    while True:
        print("Enter a message to send to BotCortex:")
        message = input()
        print("BotCortex: " + str(BotCortex.talk_history(message)))


if __name__ == "__main__":
    main()
