```mermaid
stateDiagram-v2
    [*] --> Idle_AwaitingInput

    Idle_AwaitingInput --> InputReceived_Classifying: Input Text Received

    InputReceived_Classifying --> Processing_VisualQuestion: Classified as Visual
    InputReceived_Classifying --> Processing_FollowUpNeeded: Classified as Needs Follow-up
    InputReceived_Classifying --> Processing_MemoryQuery: Classified as Question for Memory
    InputReceived_Classifying --> Processing_GeneralChat_LLM: Classified as General/Fallback

    Processing_VisualQuestion --> GeneratingResponse: VQA Result
    Processing_FollowUpNeeded --> GeneratingResponse: Follow-up Question Generated
    Processing_MemoryQuery --> GeneratingResponse: Found in Memory / Not Found (Result passed for formulation)
    Processing_GeneralChat_LLM --> GeneratingResponse: LLM Output

    GeneratingResponse --> ResponseReady: Response Finalized
    ResponseReady --> Idle_AwaitingInput: Response Sent

    state InputReceived_Classifying {
        note right of InputReceived_Classifying
            Uses helpers.py for classification:
            - is_visual_question()
            - needs_follow_up()
            - is_question() & is_personal()
        end note
    }

    state Processing_VisualQuestion {
        note left of Processing_VisualQuestion
            Involves:
            - Image capture (URL or Picoh)
            - VQA Model (e.g., GIT)
        end note
    }
    
    state Processing_FollowUpNeeded {
         note right of Processing_FollowUpNeeded
            Involves:
            - Question Generation Model
         end note
    }

    state Processing_MemoryQuery {
        note left of Processing_MemoryQuery
            Involves:
            - Semantic search on question_file.json
            - Sentence Transformer for embeddings
        end note
    }

    state Processing_GeneralChat_LLM {
        note right of Processing_GeneralChat_LLM
            Involves:
            - Primary LLM (Local or Cloud)
            - Context: History, NER, Summary, Topics
        end note
    }
```
