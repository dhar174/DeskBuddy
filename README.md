# DeskBuddy
 An intelligent chatbot with context memory, emotional awareness, and question-answering capabilities that runs on local hardware. Video coming soon.

# Current Features as of 1.26.2023 (full setup):
 1. Connects to expressive robot which syncs mouth movements with language model output ![1914_1_jpg](https://user-images.githubusercontent.com/5241499/214900213-4f4e0c39-f747-4ef7-ac8b-f32137187e3d.png)

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
