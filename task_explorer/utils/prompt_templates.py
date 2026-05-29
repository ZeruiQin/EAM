# INPUTS: app_name, package_name, activity_list
TASK_GOAL_GENERATOR = """Given the screenshot of {app_name} and its available activities, generate a comprehensive list of practical user tasks that:

1. Start from the current screen shown in the screenshot
2. Can be completed within 10-30 steps
3. Utilize the app's full feature set based on the activity list
4. Are concrete and specific (like searching for a particular item rather than just "search")
5. Cover different user interaction patterns (viewing, editing, sharing, etc.)
6. Include both basic and advanced features
7. Represent realistic user behaviors and goals
8. Avoid excessive steps on form-filling or scrolling pages

Important context:
- App name: {app_name}
- Package name: {package_name} 
- Available activities (app screens/features):
```{activity_list}```

Format requirements:
1. List only the tasks without explanations or commentary
2. Each task should be a single, clear directive
3. Use specific examples (e.g., concrete search terms, actions, settings)
4. Include the expected outcome where relevant
5. Tasks should follow this pattern: [Starting action] + [Specific steps] + [End goal]

Example tasks from other apps (for reference only):
1. Search for "ocean waves" white noise, then sort results by most played
2. Open the first recommended video, then post "Great content!" as a comment
3. Play the trending video, then add it to your "Watch Later" playlist
4. Navigate to the comments section of a featured video, then like the top comment

Generate diverse tasks that would help a user explore and utilize all major features visible in the screenshot and implied by the activity list."""

SUBTASK_GOAL_GENERATOR = """Given a user task, the current screenshot of {app_name}, and available UI elements, generate multiple potential sub-goals to progress toward completing the user task. Each sub-goal must:

1. Start with interacting with a specific UI element from the provided element list
2. Be expressed as a single, clear directive following the pattern: [Starting action] + [Specific steps] + [End goal]
3. Be achievable within approximately 3 actions(AT MOST 5) from the anchor element
4. Provide a concrete target state that advances toward the user task completion

**Context Information**:
- User Task: {user_task}
- App name: {app_name}
- Package name: {package_name}
- Current screen elements(Only interact with *visible=true elements):
```{element_list}```
- Activity context: {activity_list}
- Recent History Action(up to 5): {action_history}
- Sub-goals History: {subgoal_history}
- State Summary: {state_summary}

**Task Execution Analysis**:
Before generating new sub-goals, analyze the current execution state:

1. **Completed Progress**: Review the sub-goals history to understand what has already been accomplished toward the main task
2. **Current Position**: Based on the state summary and recent actions, identify where you are in the task workflow
3. **Remaining Work**: Determine what specific components of the user task still need to be completed
4. **Next Logical Steps**: Identify the most logical next actions that build upon completed sub-goals

**Sub-goal Generation Strategy:**
- **Continuation Focus**: Generate sub-goals that logically continue from where previous sub-goals left off
- **Avoid Redundancy**: Do not repeat actions or objectives that have already been successfully completed
- **Progressive Advancement**: Each sub-goal should represent a clear step forward in the overall task completion
- **Context Awareness**: Consider the current screen state and how it relates to the overall task progress

For each sub-goal, provide:
1. **Anchor Element**: The specific UI element ID/description from the list to start with
2. **Sub-goal**: Single directive sentence following [Starting action] + [Specific steps] + [End goal] pattern
3. **Confidence Score**: How likely this sub-goal is to advance toward task completion (0.0-1.0)

Format each sub-goal as:
Sub-goal [N]:
Anchor: [Element ID/description from element_list]
Directive: [Single clear instruction with starting action + steps + end goal]
Confidence: [0.0-1.0]



Example format (for reference):
Sub-goal 1:
Anchor: Search bar in header
Directive: Tap the search bar, enter "wireless headphones", then reach the search results page
Confidence: 0.9

Sub-goal 2:
Anchor: Categories menu button
Directive: Open the categories menu, select "Electronics", then navigate to the electronics browsing page
Confidence: 0.8



Requirements:
- Generate 3-5 different sub-goals exploring different potential paths
- Each sub-goal MUST start with a different anchor element from the provided list
- Each directive should be a single, comprehensive sentence that clearly states what to accomplish
- Sort sub-goals by confidence score (highest first)
- Focus on concrete target states that bring the user closer to task completion
- Directives should be specific enough to provide clear execution guidance but flexible enough to adapt to actual UI responses

Generate diverse exploration paths that enable the agent to try alternative approaches if the highest-confidence path fails."""


TASK_COMPLETION_CHECKER = """Given the user task, action history, and current screenshot of {app_name}, determine whether the user task has been successfully completed.

Context Information:
- User Task: {user_task}
- App name: {app_name}
- Package name: {package_name}
- Recent Action History: {action_history}
- Sub-goals History: {subgoal_history}
- Current screen elements:
```{element_list}```

Analysis Requirements:
1. Compare the current state with the expected end goal of the user task
2. Consider whether all required steps have been executed successfully
3. Verify if the current screen/state indicates task completion
4. Account for any error states or incomplete actions in the history
5. Be strict about completion - partial progress is not completion

Evaluation Criteria:
- Has the user task's primary objective been achieved?
- Are we at the expected final state/screen for this task?
- Have all necessary intermediate steps been completed?
- Is there clear evidence of successful task completion in the current state?

Format your response as:
Reasoning: [Detailed analysis of why the task is or isn't complete, referencing specific elements from the action history and current state]
Result: [True/False]

Examples:

For task "Search for wireless headphones and add the first result to cart":
Reasoning: The action history shows successful search execution and product selection, and the current screen displays "Added to Cart" confirmation with the wireless headphones item visible in the cart summary.
Result: True

For task "Change profile settings to private mode":
Reasoning: The action history shows navigation to settings and privacy options, but the current screen still shows "Public" status in the profile visibility setting, indicating the change was not successfully applied.
Result: False

Analyze the provided context and determine task completion status."""

STATE_EVALUATOR = """Given the user task, action history, and current screenshot of {app_name}, evaluate the current exploration state and determine the next action strategy.

Context Information:
- User Task: {user_task}
- App name: {app_name}
- Package name: {package_name}
- Recent Action History: {action_history} (if the last action is '{{"action_type": "status", "goal_status": "complete"}}', it means the last sub-goal was complete successfully)
- Sub-goals History: {subgoal_history}
- Current screen elements:
```{element_list}```

Analysis Requirements:
1. Compare the current state with the expected end goal of the user task
2. Evaluate whether the recent actions are leading toward task completion
3. Assess if the current exploration path is meaningful and relevant
4. Consider whether all required steps have been executed successfully
5. Verify if the current screen/state indicates task completion
6. Account for any error states, dead-ends, or repetitive actions in the history
7. Make sure to use answer action for information retrieval task({{"action_type": "answer", "text": "<answer_text>"}} is the last action in the action history)
7. Be strict about completion - partial progress is not completion

Evaluation Criteria:
- Has the user task's primary objective been achieved? (COMPLETED)
** Have we completed all the sub-goals required by the task and at the expected final state/screen for this task? 
- Are we making meaningful progress toward the goal? (CONTINUE)
- Are we stuck, going in wrong direction, or exploring irrelevant paths? (BACKTRACK)
- Is there clear evidence of task completion, progress, or deviation in the current state?

Format your response as:
Reasoning: [Detailed analysis of current progress, referencing specific elements from action history, current state, and task relevance. Explain why we should continue, backtrack, or if task is complete]
Result: [CONTINUE/BACKTRACK/COMPLETED]

Examples:

For task "Search for wireless headphones and add the first result to cart":
Reasoning: The action history shows successful search execution and product selection, and the current screen displays "Added to Cart" confirmation with the wireless headphones item visible in the cart summary. The task objective has been fully achieved.
Result: COMPLETED

For task "Change profile settings to private mode":
Reasoning: Recent actions show navigation to settings menu and we can see privacy-related options on current screen. We're moving in the right direction toward the settings area, making meaningful progress toward the goal but haven't reached the privacy settings yet.
Result: CONTINUE

For task "Send message to John":
Reasoning: The action history shows we clicked on a shopping icon instead of messaging app, and current screen shows product listings which are completely unrelated to messaging functionality. This exploration path will not lead to task completion.
Result: BACKTRACK

Analyze the provided context and determine the exploration state."""


SCREENSHOT_COMPARISON = """You are an expert UI analyzer tasked with determining if screenshots represent the SAME FUNCTIONAL PAGE STATE.

Context Information:
- Task: {goal}
- Current Action: {action_reasoning}


CRITICAL EVALUATION CRITERIA:
1. **Functional State Identity**: Pages must provide the SAME user functionality and options
2. **Content Consistency**: Core content, available actions, and user interface elements must be substantially identical
3. **Interactive Elements**: All clickable elements, buttons, toggles, and controls must be in the SAME STATE
4. **Information Display**: The same information must be visible and accessible to users
5. **Core Elements and Text**: The elements and text related to the current action must be the same

STRICT CLASSIFICATION RULES:
- **SAME PAGE**: Only when screenshots show identical functionality, same interactive element states, and same available user actions
- **DIFFERENT PAGE**: When core functionality differs, interactive elements are in different states, or available actions change significantly

IGNORE ONLY THESE MINOR DIFFERENCES:
- Timestamps, clock displays
- Network signal strength
- Battery level indicators
- Minor text content updates (notifications, counters)
- Scroll position within the same content area
- Animation states or loading indicators

DO NOT IGNORE THESE CRITICAL DIFFERENCES:
- Toggle switches in different states (ON/OFF)
- Different button states (enabled/disabled)
- Presence/absence of major UI sections
- Different available actions or menu options
- Different form states or input validation
- Different content visibility based on settings/states

EXAMPLE OF DIFFERENT PAGES:
- Bluetooth Settings with Bluetooth ON vs Bluetooth OFF
- Login page vs Dashboard page
- Empty cart vs cart with items
- Form with validation errors vs clean form
- Settings with different option availability

Please analyze the current screenshot against each candidate screenshot and ONLY return:

{{
  "is_same_page": boolean,
  "matched_candidate_index": integer or null,
  "confidence_score": float (0.0-1.0),
  "analysis_summary": "Brief explanation focusing on functional differences/similarities",
  "ignored_differences": ["List of minor differences ignored"],
  "critical_differences": ["List of functional/state differences that matter"]
}}

OUTPUT NOTICE: "matched_candidate_index" MUST NOT BE NULL if "is_same_page" is true

CURRENT SCREENSHOT: [The first image will be provided below]
CANDIDATE SCREENSHOTS: [Images will be provided following the first one]

Be STRICT in your evaluation. When in doubt about functional equivalence, classify as DIFFERENT PAGES."""


# INPUTS: task_description, numeric_tag_of_element, ui_element_attributes, action
KNOWLEDGE_EXTRACTOR = """Objective: Describe the functionality of a specific UI element in a mobile app screenshot.

Input:
- Two screenshots: Before and after interacting with a UI element
- UI element marked with a numeric tag in the top-left corner
- Element number: {numeric_tag_of_element}
- Broader task context: {task_description}
- Action taken: {action}
- UI Element Attributes: 
  ```
  {ui_element_attributes}
  ```

Requirements for Functionality Description:
1. Concise: 1-2 sentences
2. Focus on general function, not specific details
3. Avoid mentioning the numeric tag
4. Use generic terms like "UI element" or appropriate pronouns

Example:
- Incorrect: "Tapping the element #3 displays David's saved recipes in the results panel"
- Correct: "Tapping this element will initiates a search and displays matching results"

Guidance:
- Describe the core action and immediate result of interacting with the UI element
- Prioritize clarity and generality in the description"""


# INPUTS: task_goal, knowledge_a, knowledge_b
RANKER = """Given the user instruction: {task_goal}, determine which of the following two knowledge entries is more useful.
Respond ONLY with a integer value:
1 means Knowledge A is strictly better.
2 means Knowledge B is strictly better.

Knowledge A: {knowledge_a}
Knowledge B: {knowledge_b}

Please provide your response:
"""


# INPUTS: task_goal, history, ui_elements, knowledge
REASONING = """## Role Definition
You are an Android operation AI that fulfills user requests through precise screen interactions.
The current screenshot and the same screenshot with bounding boxes and labels added are also given to you.

## Action Catalog
Available actions (STRICT JSON FORMAT REQUIRED):
1. Status Operations:
   - Task Complete: {{"action_type": "status", "goal_status": "complete"}}
   - Task Infeasible: {{"action_type": "status", "goal_status": "infeasible"}}
2. Information Actions:
   - Answer Question: {{"action_type": "answer", "text": "<answer_text>"}}
3. Screen Interactions:
   - Tap Element: {{"action_type": "click", "index": <visible_index>}}
   - Long Press: {{"action_type": "long_press", "index": <visible_index>}}
   - Scroll: Scroll the screen or a specific scrollable UI element. Use the `index` of the target element if scrolling a specific element, or omit `index` to scroll the whole screen. {{"action_type": "scroll", "direction": <"up"|"down"|"left"|"right">, "index": <optional_target_index>}}
4. Input Operations:
   - Text Entry: {{"action_type": "input_text", "text": "<content>", "index": <text_field_index>}}
   - Keyboard Enter: {{"action_type": "keyboard_enter"}}
5. Navigation:
   - Home Screen: {{"action_type": "navigate_home"}}
   - Back Navigation: {{"action_type": "navigate_back"}}
6. System Actions:
   - Launch App: {{"action_type": "open_app", "app_name": "<exact_name>"}}
   - Wait Refresh: {{"action_type": "wait"}}

## Current Objective
User Goal: {task_goal}

## Execution Context
Action History:
{history}

Visible UI Elements (Only interact with *visible=true elements):
{ui_elements}

## Core Strategy
1. Path Optimization:
   - Prefer direct methods (e.g., open_app > app drawer navigation)
   - Always use the `input_text` action for entering text into designated text fields.
   - Verify element visibility (`visible=true`) before attempting any interaction (click, long_press, input_text). Do not interact with elements marked `visible=false`.
   - Use `scroll` when necessary to bring off-screen elements into view. Prioritize scrolling specific containers (`index` provided) over full-screen scrolls if possible.
   
2. Error Handling Protocol:
   - Switch approach after ≥ 2 failed attempts
   - Prioritize scrolling (`scroll` action) over force-acting on invisible elements
   - If an element is not visible, use `scroll` in the likely direction (e.g., 'down' to find elements below the current view).
   - Try opposite scroll direction if initial fails (up/down, left/right)
   - If the `open_app` action fails to correctly open the app, find the corresponding app in the app drawer and open it.

3. Information Tasks:
   - MANDATORY: Use answer action for questions
   - Verify data freshness (e.g., check calendar date)

## Expert Techniques
Here are some tips for you:
{knowledge}

## Response Format
STRICTLY follow:
Reasoning: [Step-by-step analysis covering:
           - Visibility verification
           - History effectiveness evaluation
           - Alternative approach comparison
           - Consideration of scrolling if needed]
Action: [SINGLE JSON action from catalog]

Generate response:
"""

# INPUTS: task_goal, before_ui_elements, after_ui_elements, action, reasoning
SUMMARY="""
Goal: {task_goal}

Before screenshot elements:
{before_ui_elements}

After screenshot elements:
{after_ui_elements}

Action: {action}
Reasoning: {reasoning}

Provide a concise single-line summary (under 50 words) of this step by comparing screenshots and action outcome. Include:
- What was intended
- Whether it succeeded
- Key information for future actions
- Critical analysis if action/reasoning was flawed
- Important data to remember across apps

For actions like 'answer' or 'wait' with no screen change, assume they worked as intended.

Summary:
"""