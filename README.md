### Telco Customer Multi-Agent AI System

##### Project Overview

- This project implements an end-to-end multi-agent AI system for telecom customer support using Databricks.

- The project extends a single-agent customer-support workflow into a modular multi-agent architecture in which specialized agents collaborate through a shared workflow state.

- A Coordinator Agent analyzes each user request and creates an execution plan. A dependency-based orchestrator then executes only the required specialist agents in the correct order.

The system supports:

    - SQL analytics
    - Customer churn prediction
    - Semantic search over customer notes
    - Retention recommendations
    - Grounded natural-language responses
    - Dynamic agent routing
    - Dependency-based orchestration
    - Structured validation
    - Error handling
    - Unit and end-to-end integration testing

##### Business Use Case

Telecom customer-support requests can require different types of processing.

Example requests include:

- How many customers churned?

- What is the churn rate?

- Will customer 7590-VHVEG churn?

- Why are customers likely to cancel service?

- For customer 7590-VHVEG, 
    - predict churn risk,
    - review similar customer notes,
    - recommend a retention action,
    - and provide a final response.

Rather than sending every request through the same workflow, the system dynamically determines which agents are required.

##### Final Architecture


``` text

                         User Request
                              │
                              ▼
                    ┌───────────────────┐
                    │ Coordinator Agent │
                    └─────────┬─────────┘
                              │
                              ▼
                       Execution Plan
                              │
                              ▼
                 ┌────────────────────────┐
                 │ Multi-Agent            │
                 │ Orchestrator           │
                 └────────────┬───────────┘
                              │
              Dependency-Based Execution
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌────────────────┐   ┌───────────────────┐
   │  SQL Agent  │    │ Prediction     │   │ Vector Search     │
   │             │    │ Agent          │   │ Agent             │
   └──────┬──────┘    └───────┬────────┘   └─────────┬─────────┘
          │                    │                      │
          │                    └──────────┬───────────┘
          │                               │
          │                               ▼
          │                    ┌────────────────────┐
          │                    │ Retention Agent    │
          │                    └──────────┬─────────┘
          │                               │
          └───────────────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Final Response      │
                    │ Agent               │
                    └──────────┬──────────┘
                               │
                               ▼
                    Grounded Final Response

```

The exact execution path depends on the Coordinator's plan. Not every request executes every specialist agent.

Agent Responsibilities
| Agent                    | Responsibility                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| **Coordinator Agent**    | Understands the user request and creates a structured execution plan                     |
| **SQL Agent**            | Handles counts, rates, averages, summaries, and structured analytics                     |
| **Prediction Agent**     | Predicts churn risk for a specific customer using the deployed churn model               |
| **Vector Search Agent**  | Retrieves semantically relevant customer notes, complaints, and cancellation information |
| **Retention Agent**      | Uses available customer evidence to recommend an appropriate retention action            |
| **Final Response Agent** | Generates the final grounded natural-language response from validated agent results      |



The orchestrator is not a specialist agent. It coordinates execution and does not perform SQL analytics, prediction, Vector Search, retention reasoning, or final-response generation itself.


##### Example Execution Paths

###### A SQL analytics request:


``` text

User
 ↓
Coordinator
 ↓
SQL Agent
 ↓
Final Response Agent

```

###### A churn prediction request:

``` text 

User
 ↓
Coordinator
 ↓
Prediction Agent
 ↓
Churn Model Endpoint
 ↓
Final Response Agent

```

###### A customer-notes request:

``` text 

User
 ↓
Coordinator
 ↓
Vector Search Agent
 ↓
Final Response Agent

```


###### A combined retention request:


``` text 

User
 ↓
Coordinator
 ├───────────────┐
 ↓               ↓
Prediction    Vector Search
Agent         Agent
 └───────┬───────┘
         ↓
   Retention Agent
         ↓
 Final Response Agent
         ↓
 Grounded Response

```

###### Shared Workflow State

All agents communicate through a common MultiAgentState.

Conceptually, the state contains:

``` text 

{
    "user_request": ...,
    "coordinator_result": ...,
    "agent_results": {...},
    "execution_history": [...],
    "errors": [...],
}

```

This provides one controlled location for passing information between agents.

Agents do not need to call one another directly. They read required information from shared state and store their validated results back into it.


###### Dependency-Based Orchestration

Each task generated by the Coordinator contains dependencies.

Example:

Prediction Agent
depends_on = []

Vector Search Agent
depends_on = []

Retention Agent
depends_on = [
    Prediction Agent,
    Vector Search Agent
]

Final Response Agent
depends_on = [
    Retention Agent
]

###### The orchestrator repeatedly:

Find incomplete tasks
        ↓
Check dependencies
        ↓
Identify ready tasks
        ↓
Execute ready agents
        ↓
Store validated results
        ↓
Repeat until workflow is complete

This prevents downstream agents from executing before the information they require is available.


### Project Notebook Sequence

##### Shared Components

##### 01_shared_models

Defines the common typed contracts used throughout the multi-agent system.

Includes concepts such as:

- AgentName
- AgentStatus
- RequestType
- AgentTask
- Base agent results
- Agent-specific result models
- Coordinator result
- Execution records
- Error records

Pydantic provides runtime validation of structured agent outputs.


##### 02_shared_state_and_helpers

Defines the shared workflow state and reusable helper functions.

Responsibilities include:

- Creating initial state
- Recording agent execution
- Recording errors
- Finding agent tasks
- Dependency validation
- Shared workflow utilities

This notebook provides the common infrastructure used by all agents.


#### Agent Notebooks

##### 03_coordinator_agent

Implements the Coordinator Agent.

Responsibilities:

``` text 

User Request
      ↓
Coordinator LLM
      ↓
Structured JSON
      ↓
Pydantic Validation
      ↓
Execution Plan

```

The Coordinator:

- Determines request type
- Selects required agents
- Avoids unnecessary agents
- Defines execution order
- Defines task dependencies
- Always places the Final Response Agent last


#### 04_sql_agent

Implements structured telecom analytics.

Example questions:

- How many customers churned?
- What is the churn rate?

Example result:

- sql_action = churn_rate
- churn_rate_percent = 26.54


##### 05_prediction_agent

Implements customer churn prediction.

Example:

Will customer 7590-VHVEG churn?

Workflow:

``` text 

Customer ID
     ↓
Prediction Agent
     ↓
Databricks Model Serving Endpoint
     ↓
Prediction Tool Response
     ↓
Validated PredictionAgentResult

```

Example prediction:

- Customer: 7590-VHVEG
- Prediction: Churn


##### 06_vector_search_agent

Implements semantic retrieval over customer notes.

Workflow:

``` text

Natural-Language Question
          ↓
Embedding
          ↓
Databricks Vector Search
          ↓
Similar Customer Notes
          ↓
Validated VectorSearchAgentResult

```
This enables semantic questions such as:

- Why are customers likely to cancel service?


##### 07_retention_agent

- Implements retention recommendation logic.

- The Retention Agent can consume validated upstream results such as:

``` text

Prediction Result
       +
Customer Notes
       ↓
Retention Agent
       ↓
Recommended Action

```

Example retention actions include:

- offer_discount
- offer_support_package
- service_quality_review
- billing_review
- no_action


##### 08_final_response_agent

- Generates the final grounded user-facing response.

- It uses successful validated agent results rather than independently performing specialist work.

``` text

Validated Agent Results
        ↓
Grounding Context
        ↓
Final Response LLM
        ↓
FinalResponseAgentResult

```
Failed agent results are excluded from grounding.


### Orchestration

##### 09_end_to_end_multi_agent_orchestration

Connects all agents into one dependency-driven workflow.

The orchestrator:

- Creates the initial state
- Runs the Coordinator
- Reads the execution plan
- Determines which tasks are ready
- Checks dependencies
- Executes agents
- Stores results
- Records execution history
- Records workflow errors
- Detects blocked workflows
- Protects against excessive iterations
- Stops when all planned tasks have completed

The orchestrator coordinates the workflow but does not perform specialist business logic itself.


### Testing

##### 10_end_to_end_integration_test

- Validates the complete system using real integrated components.

- The project includes both unit testing and end-to-end integration testing.

- Unit tests validate individual agent/orchestration behaviors, including:

    - Dependency ordering
    - Blocked workflows
    - Maximum-iteration protection
    - Agent routing
    - Validation behavior
    - Error handling

- End-to-end tests validate complete business workflows.

Examples include:

    - SQL Analytics Workflow
    - Prediction Workflow
    - Vector Search Workflow
    - Retention Workflow
    - Churn Rate Workflow

- A successful churn-rate workflow demonstrates dynamic routing:

``` text 

"What is the churn rate?"
          ↓
Coordinator
          ↓
SQL Agent
          ↓
Final Response Agent
          ↓
"The churn rate is 26.54%."

```

No Prediction, Vector Search, or Retention Agent is required for that request.


### Technologies Used

- Python
- Databricks
- Databricks Model Serving
- Databricks Vector Search
- Unity Catalog / Delta tables
- MLflow
- Large Language Models
- Embeddings
- Pydantic
- TypedDict
- Python type hints
- Dependency injection
- Shared-state orchestration
- Unit testing
- Integration testing


### Key Design Principles

##### Separation of Responsibilities

   Each agent owns one clearly defined responsibility.

##### Dynamic Routing

The Coordinator determines which agents are actually necessary for each request.

##### Dependency Injection

LLM invocation functions and deterministic tools are injected into agents and orchestration functions rather than being tightly coupled to implementations.

##### Structured Validation

Pydantic models provide contracts between:

- Coordinator
- Agents
- Tools
- Shared State
- Orchestrator

##### Grounded Generation

The Final Response Agent uses validated agent results as grounding context.

##### Explicit Error Handling

Failures are recorded consistently in shared state and execution history.

##### Testability

Individual components can be tested independently before validating the complete integrated system.


### Key Learnings

This project provided practical experience with:

- Multi-agent architecture
- Coordinator routing
- Dependency injection
- Shared-state management
- Pydantic validation
- Tool integration
- Tool-response validation
- Dependency-based orchestration
- Grounded LLM responses
- Error handling
- Debugging multi-component AI systems
- LLM token/output limitations
- Databricks notebook namespace and %run behavior
- Unit testing
- End-to-end integration testing

A particularly important learning was that successful individual components do not guarantee a successful multi-agent workflow.


Integration exposed issues involving:

- LLM output truncation
- Malformed JSON
- Prompt design
- Tool contracts
- Customer-ID extraction
- Incorrect dependencies
- Notebook namespaces
- Function redefinition
- Execution ordering
- 
Debugging these interactions was an important part of understanding how production-style multi-agent systems behave.


### Conclusion

- This project implemented an end-to-end multi-agent AI system for telecom customer support using Databricks.

- The project defined the multi-agent architecture and established clear responsibilities for the Coordinator, SQL, Prediction, Vector Search, Retention, and Final Response Agents.

- Validated input and output schemas were implemented using Pydantic, together with a shared workflow state and consistent execution-history, dependency, status, and error-handling conventions.

- The Coordinator dynamically creates execution plans based on the user's request, while the orchestrator executes only the required agents according to their dependencies.

- Each specialist agent integrates with its corresponding tool, validates the tool response, and stores its result in shared state. The Final Response Agent produces a grounded response using validated results from successfully completed agents.

- Finally, unit and end-to-end integration testing verified routing, dependency ordering, tool integration, workflow completion, error handling, and final-response generation.


The project demonstrates how specialized AI agents, deterministic tools, shared state, structured validation, and dependency-based orchestration can work together to build a modular and reliable multi-agent AI application.
