# DeskBuddy
 An intelligent chatbot with context memory, emotional awareness, and question-answering capabilities that runs on local hardware. Cooper stands for Conversational Office Organizer and Personal Entertainment Robot.
 
# Introducing C.O.O.P.E.R.: Conversational Office Organizer and Personal Entertainment Robot!


## Install and validate the portable baseline

Use **Python 3.11 or 3.12** and [uv](https://docs.astral.sh/uv/getting-started/installation/)
0.9.26 or newer. On Windows, install uv with `winget install --id=astral-sh.uv -e`;
on Linux/macOS use the official installer linked above. From a clone of this repository:

```console
uv sync --locked --no-editable
uv run --no-sync deskbuddy --headless
```

The first command installs the locked core as an ordinary package. It requires no
local package checkout outside this repository, SSH access, CUDA, model weights,
robot, microphone or camera. The second command validates settings and prints a
small status report; it starts no network connections or models. `python -m deskbuddy`
works in the installed environment too. To validate through the historical entry
point: `uv run --no-sync python start_bot.py --headless`.

The `src/deskbuddy` package is the incremental configuration/bootstrap boundary.
The legacy GUI and inference modules remain at the repository root for later
architecture work (#7). A core install proves configuration and importability;
the full 2023 robot demonstration still requires the optional subsystems described below.

### Configuration

Copy `.env.example` to `.env`, change the settings you need, then run:

```console
uv run --no-sync deskbuddy --env-file .env
uv run --no-sync deskbuddy --config deskbuddy.example.toml
```

Precedence is **explicit Python overrides > process environment > explicitly selected
dotenv file > TOML file > defaults**. `.env` is loaded only with `--env-file` or
`DESKBUDDY_ENV_FILE`; it is never searched for implicitly. A TOML file can be passed
with `--config`/`DESKBUDDY_CONFIG_FILE`; otherwise the OS user config directory's
`DeskBuddy/deskbuddy.toml` is used if present. TOML keys live under `[deskbuddy]`.
For the legacy scripts, set these environment file selectors before starting them.
Neither source loading nor the headless command creates files or directories.

Every field uses `DESKBUDDY_<FIELD_NAME>` except the supported OpenAI credential,
which uses **`OPENAI_API_KEY`**. The old generic `API_KEY` is ignored. OpenAI mode
requires a nonblank key; disabled/local mode requires none. Defaults use loopback
robot addresses, port 8888, no camera and automatic device/memory selection.
The examples document all fields, including the robot authentication/timeouts
contract for #4. Invalid values fail before startup with an actionable message.

Application data and cache use [platformdirs](https://platformdirs.readthedocs.io/)
OS conventions, not the checkout. Relative `data_dir` is under the OS default data
directory; other relative setting paths (including paths from TOML) are under the
resolved `data_dir`. File selector paths passed to the CLI are relative to the
caller. See [setup and migration details](docs/setup.md) for locations and old-data migration.

### Optional capabilities and development

Install only the extras you need; repeat `--extra` to combine them:

```console
uv sync --locked --no-editable --extra cloud
uv sync --locked --no-editable --extra local --extra vision
uv sync --locked --no-editable --extra speech --extra picoh
uv sync --locked --group dev
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync mypy
```

| Extra/group | Purpose and prerequisites |
| --- | --- |
| core (default) | Settings and offline bootstrap only; no GPU or vendor libraries |
| `cloud` | OpenAI 0.28.1 SDK compatibility with existing API calls; no local weights |
| `local` | Torch/Transformers/Accelerate and tokenizers; large wheels, opt-in downloads |
| `speech` | Whisper/PyAudio/keyboard/audio support; FFmpeg and PortAudio may need OS installation |
| `vision` | Headless OpenCV and Pillow; no automatic model or camera access |
| `picoh` | Windows vendor integration; physical hardware/resources required for actual use |
| `legacy` | Experimental Linux DeepSpeed, spaCy, datasets and classifier compatibility |
| `dev` group | pytest, pytest-asyncio, Ruff and mypy; never part of the default core install |

`local_model_endpoint` establishes configuration for a future endpoint adapter;
this baseline does not introduce an inference server client. Selecting `local`
in bootstrap validates configuration and does not start inference. Legacy GUI cloud
mode still uses local classifiers/embeddings and is not a cloud-only runtime.

For an API-only diagnostic without local weights, install `--extra cloud`, set
`DESKBUDDY_MODEL_PROVIDER=openai`, `OPENAI_API_KEY`, and `DESKBUDDY_OPENAI_MODEL`
to a model available on your account, then run `uv run --no-sync deskbuddy --prompt
"Hello"`. Supplying `--prompt` explicitly sends one API request and may incur API
charges. The diagnostic has no robot actions, classifiers or conversation history;
ordinary headless validation makes no API call. Offline tests mock this request.

The legacy GUI is launched with `uv run --no-sync python start_bot.py` only after
installing its documented extras and model assets. `activateBuddy.ps1 -Headless`
is portable and safe for baseline validation. The default PowerShell launcher is
explicitly legacy and starts the Windows robot server. Its transport and the Unix
client retain the old RPC implementation until #4 lands; secret/timeout fields
are a shared contract, with enforcement owned by #4.

The old `requirements.txt` freeze has been replaced with a generated, hash-pinned
**core dependency export**; uv plus `uv.lock` is the development authority. Do not
reuse an old virtualenv or `windows/buddyreqs.txt`. The [dependency audit](docs/dependency-audit.md)
records every old dependency and the [migration guide](docs/setup.md) explains
Python compatibility, legacy limits and the unchanged Larynx Debian Dockerfile.

## Historical replication disclaimer (2023 full robot demo)

I cannot guarantee this will run for you on your own machine. This is still an alpha project, and I plan to provide better instructions for replication as well as a graphical model replacement of the robot (a 3D face, basically) so the robot will not be required to run this. Also, with the online mode, it may run on more diverse hardware. Also, the Larynx docker server is customized by me; it is almost ready to place here and is coming soon as of May 2 2023. These scripts will not run without my custom Larynx server scripts; as of May-3-23 I am more focused on developing COOPER and adding features than worrying about reproducibility and it probably won't run for you. But a version everyone can use IS COMING!!! :-)

### **Videos of COOPER in action are at the bottom of the page!**

## Update May 02 2023:
Fixed COOPER's lip syncing! It is not perfect. But it does sync to a good degree, mostly accurately! I think the limitations come from the robot hardware- the lip sync starts off well aligned and gets worse as it speaks longer (the longer the response, the more apparent this is)
I have also added an online mode - the offline mode was always intended as a backup or alternative to spending money on API calls (and will continue being developed further as an important part of COOPER). COOPER now uses the gpt-3.5-turbo model using the OpenAI API for much improved intelligence and response times. Make no mistake about it though - COOPER is still COOPER, with the same cheeky attitude!



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

### If anyone would like to assist me by providing material compute resources, please contact me. I will happily accept donations of hardware


# Authenticated Robot Transport Protocol

DeskBuddy uses a length-framed, HMAC-SHA256 authenticated, typed JSON transport between `start_bot.py` (`robot_client.py`) and `buddy1.py` (controlling the Ohbot Picoh robot). This replaces the legacy insecure `pickle` RPC over TCP.

## Architecture & Wire Protocol

- **Framing**: 4-byte big-endian unsigned integer length prefix followed by UTF-8 encoded JSON payload:
  `[4-byte header (big-endian uint32)][JSON payload bytes]`
- **Serialization**: Non-executable JSON serialization (`json.dumps`/`json.loads`). No Python object pickling, eval, or exec.
- **Envelope Schemas**:
  - `RobotRequest`:
    - `request_id`: Unique identifier (UUID4 string)
    - `protocol_version`: `"1.0"`
    - `timestamp`: Unix epoch float seconds
    - `command`: Command name string
    - `arguments`: JSON dictionary of arguments
    - `auth`: `{"scheme": "hmac-sha256", "signature": "<hex-digest>"}`
  - `RobotResponse`:
    - `request_id`: Correlated request identifier (or null if unparseable)
    - `protocol_version`: `"1.0"`
    - `ok`: Boolean status flag
    - `result`: Command output on success
    - `error`: Structured error object `{"code": "<error_code>", "message": "<sanitized_detail>"}` on failure

## Machine-Readable Error Codes

- `malformed_request`: Invalid framing, unparseable JSON, or missing required envelope fields.
- `unsupported_protocol`: Protocol version not recognized by the server.
- `unauthorized`: Missing, invalid, or tampered HMAC signature.
- `expired_request`: Timestamp exceeds the configured clock skew threshold.
- `replayed_request`: Request identifier has already been processed within the validity window.
- `unsupported_command`: Command is not in the explicit allowlist.
- `invalid_arguments`: Command arguments fail schema, type, range, or length validation.
- `payload_too_large`: Declared frame length exceeds the configured maximum before allocating memory.
- `timeout`: Network read, write, connect, or request deadline exceeded.
- `internal_error`: Sanitized handler error during command execution.

## Symmetric Authentication & Replay Protection

Communication between the DeskBuddy client and robot server is cryptographically signed using HMAC-SHA256 with domain separation:
- **Requests** are signed using the `deskbuddy-request-v1` domain:
  ```
  deskbuddy-request-v1|{protocol_version}|{request_id}|{timestamp:.6f}|{command}|{canonical_arguments_json}
  ```
- **Responses** are signed using the `deskbuddy-response-v1` domain:
  ```
  deskbuddy-response-v1|{protocol_version}|{request_id}|{timestamp:.6f}|{status}|{canonical_result_json}|{canonical_error_json}
  ```
  where `status` is `"1"` on success and `"0"` on failure.

Key security guarantees:
- Domain separation prevents cross-protocol reflection attacks (requests cannot be replayed as responses or vice versa).
- Constant-time HMAC comparison prevents timing attacks (`hmac.compare_digest`).
- Bounded clock drift window (default: 30s) prevents replay of expired messages.
- Server-side TTL-aware replay cache (`ReplayCache`) prevents duplicate command execution.
- **Fail Closed on Startup**: The server strictly requires a non-empty shared secret (`robot_auth_secret`) and will abort startup before binding to a network port if unconfigured.
- **HMAC Scope & Confidentiality**: HMAC-SHA256 guarantees authenticity and message integrity, preventing unauthorized command injection, tampering, and replay. However, HMAC **does not encrypt traffic payloads**. For multi-host deployments over non-local networks, traffic must be routed over a trusted private network, VPN, or SSH tunnel to ensure confidentiality.
- **Ambiguous Execution Semantics (`ExecutionOutcomeUnknown`)**:
  - A network timeout or connection reset after a command frame was transmitted does **not** prove that a physical robot action did not execute.
  - Distinguishes definitively unsent requests (ordinary `TransportError`) from uncertain execution (`ExecutionOutcomeUnknown`).
  - `ExecutionOutcomeUnknown` is raised whenever:
    1. A connection drops after a side-effecting frame may have reached the socket.
    2. A socket write or drain partially succeeds and then fails.
    3. Read times out after a request was transmitted.
    4. Overall request deadline expires after transmission.
    5. Reconnection fails after an earlier transmission may have reached the server.
    6. A retried request receives `replayed_request` or `execution_outcome_unknown`.
  - **No Blind Retries**: Side-effecting physical actions (`move`, `say`, `reply`, `base_col`, `wait`) must **never** be automatically retried blindly when execution outcome is uncertain, to avoid hazardous repeated motor motions or speech loops.
  - **Caller Handling**: Callers receiving `ExecutionOutcomeUnknown` should identify the command and request ID, log an actionable warning, notify the operator/user if appropriate, and avoid immediate re-dispatch without confirming physical state.

## Concurrency & Hardware Timeout Contracts

1. **Client Lock Contention & Timestamp Freshness**:
   - `RobotClient.request()` serializes socket I/O across concurrent coroutines via an internal lock.
   - Request creation and HMAC signing occur immediately before transmission, *after* acquiring the client lock. This ensures queued requests waiting behind long-running operations do not become stale or get rejected with `expired_request`.
   - Time waiting for the client lock has its own bounded queueing deadline. If a request times out while waiting for the lock, it was definitively never transmitted.

2. **Timeout Hierarchy & Network Grace**:
   - **Server Execution Budget**: `compute_command_timeout(command, args, base_timeout)` dynamically scales for speech and wait durations.
   - **Client Response Budget**: `server_command_timeout + bounded_network_grace` ensures the server has adequate margin to finish execution, sign, and transmit the response frame before the client deadline expires.
   - **Connection Establishment**: Governed by `robot_connect_timeout_seconds`.
   - **Per-Frame I/O**: Governed by bounded write/read deadlines.
   - **Runtime Overrides**: Validated strictly before transmission. Zero, negative, NaN, infinite, boolean, and non-numeric values are rejected immediately with typed errors (`TypeError`, `ValueError`).

3. **Synchronous Hardware Execution & Cancellation Limits**:
   - Physical Picoh driver operations (`picoh.move`, `picoh.say`, `picoh.setBaseColour`) execute in a dedicated, serialized single-worker hardware executor (`HardwareExecutor`).
   - Cancelling a coroutine awaiting hardware execution cannot forcibly terminate active C-extension or Python thread driver routines.
   - If a synchronous operation is still running on the worker thread when the server coroutine times out, the server explicitly reports `execution_outcome_unknown` rather than falsely claiming the operation was cancelled or stopped.
   - Conflicting physical commands dispatched while a prior hardware operation remains unresolved are rejected, preserving serialized safety and preventing concurrent hardware access.
   - Queued executor operations that are cancelled before worker pickup are marked `CANCELLED` and never touch physical hardware.

## Command Allowlist & Validation

Unrestricted dynamic dispatch (`getattr(MyFunctions, ...)`) has been completely removed. Remotely invocable commands are strictly bounded:

| Command | Arguments | Validation Rules | Description |
| :--- | :--- | :--- | :--- |
| `check_for_response` | `{}` | No arguments accepted | Polls for latest Whisper STT transcription |
| `reply` | `{"message": str}` | String, length $\le 10,000$ | Commands Picoh to speak assistant response |
| `simple_test` / `ping` | `{}` | No arguments accepted | Health check / connectivity ping |
| `base_col` / `baseCol` | `{"r": int, "g": int, "b": int}` | Integers in `[0, 10]` (rejected with `invalid_arguments` if out of bounds) | Sets base LED color |
| `say` / `picoh_say` | `{"text": str}` | String, length $\le 10,000$ | Direct Picoh speech synthesis |
| `move` / `picoh_move` | `{"motor": str\|int, "position": float}` | Motor index 0–6 (or valid motor name) and position in `[0.0, 10.0]` (rejected with `invalid_arguments` if out of bounds) | Servomotor movement with physical safety limit |
| `wait` / `picoh_wait` | `{"seconds": float}` | Duration in `[0.0, 60.0]` (rejected with `invalid_arguments` if out of bounds) | Pauses Picoh execution |

## Safe Binding & Network Exposure

- **Default Binding**: The server binds to `127.0.0.1` (loopback) by default. It must NOT be exposed directly to the public Internet.
- **LAN Multi-Host Deployment**: When DeskBuddy and Picoh run on separate machines across a local network:
  - Configure `DESKBUDDY_ROBOT_BIND_ADDRESS` (or `DESKBUDDY_ROBOT_HOST`) to the server's private LAN IP on the host running `buddy1.py`.
  - Ensure traffic is routed over a secure private network, VPN, or SSH tunnel (HMAC provides authentication and integrity, not encryption).
  - Set a cryptographically strong shared secret via `DESKBUDDY_ROBOT_AUTH_SECRET`.

## Configuration Parameters

All transport parameters derive from canonical DeskBuddy settings (`deskbuddy.settings.Settings`) or explicit `TransportConfig`:

| Setting Field | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `robot_host` | `DESKBUDDY_ROBOT_HOST` | `127.0.0.1` | Target robot host address for clients |
| `robot_bind_address` | `DESKBUDDY_ROBOT_BIND_ADDRESS` | `127.0.0.1` | Network interface IP to bind server socket |
| `robot_port` | `DESKBUDDY_ROBOT_PORT` | `8888` | TCP port for robot RPC |
| `robot_auth_secret` | `DESKBUDDY_ROBOT_AUTH_SECRET` | `""` (required) | Shared secret for HMAC-SHA256 authentication |
| `robot_connect_timeout_seconds` | `DESKBUDDY_ROBOT_CONNECT_TIMEOUT_SECONDS` | `5.0` | Socket connect timeout in seconds |
| `robot_timeout_seconds` | `DESKBUDDY_ROBOT_TIMEOUT_SECONDS` | `5.0` | Base request deadline in seconds (dynamically scales with `wait`/`say`) |

## Migration from Legacy Pickle RPC

1. Legacy unpickling has been completely removed for security reasons across all robot server entrypoints (`buddy1.py`, `windows/buddy1.py`, and `windows/buddy2.py`). Unauthenticated or raw pickle payloads will be rejected with `malformed_request` or framing errors.
2. Existing application code using `robot_client.request_to_server(**kwargs)` remains fully compatible:
   - `request_to_server(function_name="check_for_response")` automatically translates to the authenticated `check_for_response` command.
   - `request_to_server(function_name="reply", message=reply)` translates to the authenticated `reply` command.
3. For new code, prefer the typed client interface:
   ```python
   from deskbuddy.settings import load_settings
   from transport import RobotClient, TransportConfig

   settings = load_settings()
   config = TransportConfig.from_settings(settings, role="client")
   async with RobotClient(config) as client:
       transcription = await client.check_for_response()
       await client.reply("Hello from DeskBuddy!")
   ```

## Running Transport Tests

Unit, negative, and integration tests do not require physical robot hardware (safe mocks and ephemeral ports are utilized):
```bash
python -m pytest tests/ -v
```





# Video Demonstrations

## Videos from May 02 update:
[![Watch the first May02 video:](https://i.ytimg.com/vi/yGFjkPFfEbI/hqdefault.jpg)](https://youtu.be/yGFjkPFfEbI)
[![Watch the second May02 video:](https://i.ytimg.com/vi/Rr-90R9IxZg/hqdefault.jpg)](https://youtu.be/Rr-90R9IxZg)
[![Watch the third May02 video:](https://i.ytimg.com/vi/x-t9CHeG57U/hqdefault.jpg)](https://youtu.be/x-t9CHeG57U)
[![Watch the fourth May02 video:](https://i.ytimg.com/vi/dnNIkHksL-g/hqdefault.jpg)](https://youtu.be/dnNIkHksL-g)

##  Older Videos

He can see!: 

[![He can see](https://i.ytimg.com/vi/yNhLSMlBzHo/hqdefault.jpg)](https://youtu.be/yNhLSMlBzHo)

Long Demo Video: 

[![Watch the demo video](https://i.ytimg.com/vi/H64mSGG_VSI/hqdefault.jpg)](https://youtu.be/H64mSGG_VSI)

Knowledge Retrieval:

[![Knowledge retrieval](https://i.ytimg.com/vi/jCFAuBvSspw/hqdefault.jpg)](https://youtu.be/jCFAuBvSspw)

Unexpected Behaviors:

[![Unexpected Behavior](https://i.ytimg.com/vi/cHEibvkdpas/hqdefault.jpg)](https://youtu.be/cHEibvkdpas)

