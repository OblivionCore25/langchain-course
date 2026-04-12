# Load environment variables from .env (e.g. LANGSMITH_API_KEY, OLLAMA host)
from dotenv import load_dotenv

load_dotenv()

# init_chat_model: a helper that creates any LangChain-supported LLM by name + provider
from langchain.chat_models import init_chat_model

# @tool: a decorator that turns a plain Python function into a LangChain Tool.
# The LLM can "call" these tools by name when it needs information.
from langchain.tools import tool

# Message types that form the conversation history:
#   SystemMessage  - instructions/persona given to the LLM
#   HumanMessage   - the user's input
#   ToolMessage    - the result returned after a tool is executed
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# @traceable: sends execution traces to LangSmith for debugging & monitoring
from langsmith import traceable

# Safety cap — prevents the agent from looping forever
MAX_ITERATIONS = 10

# ---------------------------------------------------------------------------
# MODEL SELECTION — uncomment one block to switch providers
# ---------------------------------------------------------------------------

# Option A: OpenAI (requires OPENAI_API_KEY in .env)
# from langchain_openai import ChatOpenAI  # optional direct import
MODEL_NAME = "gpt-4o-mini"
MODEL_PROVIDER = "openai"

# Option B: Gemini (requires GOOGLE_API_KEY in .env)
# MODEL_NAME = "gemini-2.0-flash"
# MODEL_PROVIDER = "google_genai"

# Option C: Local via Ollama (no API key, slower on CPU)
# MODEL_NAME = "llama3.2:3b"   # pull with: ollama pull llama3.2:3b
# MODEL_PROVIDER = "ollama"


# ---------------------------------------------------------------------------
# TOOLS
# These are the "actions" the LLM is allowed to take.
# The docstring is critical — the LLM reads it to decide when to use the tool.
# ---------------------------------------------------------------------------

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog. Generic terms like 'laptop' are accepted."""
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1200.00, "mouse": 25.00, "keyboard": 75.00}
    return prices.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on a tier.
    Available tiers: bronze, silver, gold"""
    print(f" >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discounts = {"bronze": 0.05, "silver": 0.10, "gold": 0.15}
    discount = discounts.get(discount_tier, 0.0)
    return round(price * (1 - discount), 2)

# ---------------------------------------------------------------------------
# AGENT LOOP
# This is the core pattern: LLM → tool call → result → LLM → ... → final answer
# ---------------------------------------------------------------------------

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    # Register tools in a list (for binding to the LLM) and a dict (for fast lookup by name)
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    # Create the LLM. temperature=0 means deterministic, no creativity — good for tool use.
    llm = init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=0.0)

    # bind_tools tells the LLM about the available tools.
    # Under the hood, LangChain serializes the tool schemas and passes them to the model.
    llm_with_tools = llm.bind_tools(tools)

    # The conversation history — starts with a system prompt + the user's question.
    # Every iteration we'll append the AI's response and the tool result.
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. Never guess or assume prices. ALWAYS call get_product_price first.\n"
                "2. Only call apply_discount AFTER get_product_price returns a price.\n"
                "   Pass the exact price returned — do NOT modify it.\n"
                "3. NEVER calculate discounts yourself using math.\n"
                "4. Always use the tools. Do not rely on external knowledge.\n"
                "5. If the user doesn't specify a discount tier, ask — do NOT assume one. \n"
                "6. If the user asks for a generic product (like 'laptop' or 'mouse'), do not ask for clarification. \n"
                "Just pass the generic term directly into the get_product_price tool."
            )
        ),
        HumanMessage(content=question),
    ]

    print(f"Question: {question}")
    print("=" * 60)

    # --- The agent loop ---
    # Each iteration: send messages to LLM → check if it wants to call a tool
    # If yes: run the tool, add the result to messages, loop again
    # If no: the LLM has enough info to answer → return the final answer
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Send the full conversation history to the LLM
        ai_response = llm_with_tools.invoke(messages)

        # tool_calls is a list of tools the LLM wants to invoke this turn
        tool_calls = ai_response.tool_calls

        # No tool calls = the LLM has enough info and is giving a final text answer
        if not tool_calls:
            print("No tool calls detected. Assuming final answer.")
            return ai_response.content

        # Process the first tool call (the LLM usually calls one at a time)
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")   # e.g. "get_product_price"
        tool_args = tool_call.get("args", {})  # e.g. {"product": "laptop"}
        tool_id = tool_call.get("id")       # unique ID that links this call to its result

        print(f"Tool call: {tool_name} with args: {tool_args}")

        # Look up and execute the actual Python function for this tool
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict")

        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        # Add the AI's message (with the tool call request) to history
        messages.append(ai_response)

        # Add the tool result to history so the LLM can see what the tool returned.
        # tool_call_id links this result back to the specific tool_call that requested it.
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_id)
        )

    # If we exit the loop without a final answer, something went wrong
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Starting agent...")
    result = run_agent("What is the price of a laptop with a gold tier discount?")
    print("\n--- Final Answer ---")
    print(result)