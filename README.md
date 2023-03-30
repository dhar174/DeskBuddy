# DeskBuddy
 An intelligent chatbot with context memory, emotional awareness, and question-answering capabilities that runs on local hardware. 
 
# Introducing C.O.O.P.E.R.! 
 
 Demo Video: 
[![Watch the demo video](https://i.ytimg.com/vi/H64mSGG_VSI/hqdefault.jpg)](https://youtu.be/H64mSGG_VSI)

Knowledge Retrieval:

[![Knowledge retrieval](https://i.ytimg.com/vi/jCFAuBvSspw/hqdefault.jpg)](https://youtu.be/jCFAuBvSspw)

Unexpected Behaviors:

[![Unexpected Behavior](https://i.ytimg.com/vi/cHEibvkdpas/hqdefault.jpg)](https://youtu.be/cHEibvkdpas)


# Current Features as of 1.26.2023 (full setup):
 1. Connects to expressive robot which syncs mouth movements with language model output 
 ![1914_1_jpg](https://user-images.githubusercontent.com/5241499/214901149-a21d500d-7cf4-4355-add9-1259e847ce6f.png)



 2. Robot speaks using Larynx running in a Docker container. [Github link](https://github.com/rhasspy/larynx)
 3. Robot intuitively listens for speech and does it's best to take turns. Speech recognition is done using Whisper ASR model(s) by OpenAI running in a Docker container. [Github link](https://github.com/openai/whisper)
 4. Main Language Model: Using the Pygmalion-6B model with 6-billion parameters. Inference working (albeit slowly) on single-node (desktop PC with 64GB RAM+   RTX 2080 Super 8GB VRAM) by utilizing Deepspeed Zero-3 inference with parameter offloading augmented by Huggingface Accelerate launcher. 
 5. Bot has several internal summarizers (which will be better utilized in future updates) for the purpose of assisting in understanding conversational        context. Very basic context management currently. Current ability includes topic identification and conversation summarization.
 6. Multiple model size choices for efficient utilization on a variety of hardware. This will receive many changes in upcoming updates.

# Feature and Update Roadmap, Definites (in order of priority):
 1. More efficient hardware utilization!!
 
 2. A complex and swiftly evolving plan for context understanding and real-time synthesis of contextual information. A VERY preliminary idea-map can be seen   [here](https://github.com/dhar174/DeskBuddy/blob/main/assets/User_Input_Prompt.png) ![Idea Map for Context Integration](https://github.com/dhar174/DeskBuddy/blob/main/assets/User_Input_Prompt.png)
 
  This will include an integration of a variety of low-compute classifier steps. Not only will this allow for better understanding of context and the ability to gauage appropriateness of responses, it will also allow for performance improvement by passing certain narrow-domain tasks to much smaller fine-tuned text generation models. 
  New classification steps which may lead to different models or different prompting may include:
    a. Sentiment analysis of input
    b. Sentiment analysis of output
    c. Emotional classification of output
    d. Emotional classification of output
    e. Intent prediction of input, including task identification (tasks to be defined later)
    f. Statement vs Question classification
    g. Natural language inference of outputs (and possibly inputs).
    h. Check for math problems in the input
    i. Emotional analysis of audible speech inputs
    j. Checking outputs for toxicity and appropriateness
    k. Identify relations between entities, topics
    l. Identify user sentiment about entities, topics

 3. Generating questions based on inputs or context. This will be triggered by various things, such as emotion detection, task commands that require clarification, and hypothesis consistency as determined by natural language inference models.
 
 4. Image generation on command, using Stable-Diffusion based on user input prompts as enhanced by SD prompt generation models. To be displayed on device screen.
 
 5. Face tracking. The ability to follow a face as the robot speaks to the user. This would ideally include visual speaker identification and differentiation between speakers. This will be a work in progress.
 
 6. Object recognition. Will be able to visually identify and verbally confirm object names.
 
 7. Visual action detection, for identification of user actions, to support context understanding.
 
 8. Image captioning models for visual context understanding.
 
 9. Facial recognition for recognition of specific users.
 
 10. Visual question answering. The ability to answer questions posed by users using vision-language models.
 
 11. Detect and solve math problems, even when stated as word problems.
 
 12. Search the Internet.
 
 13. Home control tasks.
 
 14. Ability to generate and tell stories.
 
 15. Ability to play a variety of games (as yet undetermined, one possibility is chess). Mad libs? Hangman? Trivia?
 
 16. A variety of fine-tuned expert models for various narrow-domain functions.


  Better model choices in general
 
  8-bit quantization (of at least some models) to reduce memory footprint
 
 # Feature and Update Roadmap, Possibles:
 1. Better speech separation approach and detection of speech (not speech recognition - Whisper does great!
 2. Better recognition of different users by speech.
 2. Secondary "persona/style" model for output enhancement
 3. Distill larger model into smaller models.
 4. Distributed hardware, possibly single-board AI computers
 
 
 # Update 3/4/2023:
 ## Major changes:
   - Added a query database to enable quick access to previously asked or common questions with known answers  
   - Input Classification. Inputs are now being classified by various levels of classification, including the following:
                    1. Is it a question?
                    2. If not a question, is it a command or just casual chat? (commands not yet implemented)
                    3. If a question, is it a personal question with no right answer or is it a fact/knowledge based question?
                    4. If a question and fact-based, is there a similar Q&A in the query database?
                    5. Is it a visually-based question?
   - As indicated above, I have added visual question answering. The bot utilizes a webcam being served as an IP cam as visual 
      input to a visual question answering model (GIT-large) (tried ViLT and it was very inaccurate).
   - Added the ability to determine if a followup-question is needed to understand the input. 
   - Added the ability to generate followup questions based on context
   - Changed the Medium model from blenderbot to pygmalion-350m (might move this one to the smaller model and use pygmalion-2.7b for medium) (I am also looking into the new LLaMA models)
   - Set up a TCP server for the Picoh robot to send transcribed inputs to the model(s) in BotCortex and retrieve the responses
   - Many minor changes, model.generate() argument changes, added contextual & history abilties to medium model as well, added processing times to debug prints
  - Added start_bot.py to start the process
  - Improved efficiency by using asynchronous functions

## Still To Be Implemented: 
- Online Web Search.
- Expansion of context database to include data curated and categorized into facts about the main user, the bot itself, and other entities (meaning not just people but any concept, ie as used in relational database terminology) as well as the relations between them. I want to keep the implementation of this simple, but if it comes down to actually using SQL or similar, then I will. These databases will include facts about entities regarding: a. Their relations to other entities b. their relations to particular topics c. the sentiments and emotions associated between the entity and other entities or topics
- Emotional and semantic classification of inputs
- Safety filters and output sanitization
- Actually implementing the hierarchical memory network for prompt-tuning???
- Improved NER and topic extraction, and better parsing of the who/what/when/where/how/why aspects of context
- Add command structure. Commands will be of several types, including: a. IoT/smart home commands, from lights etc to calendar scheduling. b. Commands for operating the PC itself c. Generative commands, ie commands to generate particular content (including image generation on demand using Stable Diffusion)
- Add "Productivity Sponsor" features, such as analysis of current screen activity to prompt user to remain productive (this feature is already at 80% completion).

## Future Experimental Features/Additions: 
      - Internal Dialogue: While currently compute resources force this one onto a backburner, in the future I would like to experiment with giving the bot it's own internal (context-aware) dialogue. According to my research, this may produce a convincing simulation of real-time awareness (assuming the compute is sufficient to run in real time!) 
     - Larger Models: Currently, my compute availability limits development involving the use of larger models. However, I would very much like to make this system scalable to any level of (minimally sufficient) hardware and cloud VMs so that capability can scale (in an automatic way) with the available hardware. So that the larger the hardware it is plugged into, the more intelligent the bot will be.

### If anyone would like to assist me by providing material resources, please contact me. I will happily accept donations of hardware. Funding would also be welcome, though I am not directly seeking it. However, it would free me up for this project full-time.
