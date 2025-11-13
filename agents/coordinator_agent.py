"""
Coordinator Agent - Uses IBM watsonx.ai for intelligent agent orchestration
"""

import json
from typing import Dict, List, Any
from pathlib import Path

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

# Environment variables
from config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    COORDINATOR_ID,
    COORDINATOR_PARAMETERS
)

from agents.sentiment_agent import SentimentAgent
from agents.creative_agent import CreativeAgent
from agents.predictive_agent import PredictiveAgent


class CoordinatorAgent:
    """
    Coordinator that uses IBM watsonx.ai to intelligently orchestrate specialist agents.
    Instead of keyword matching, watsonx.ai decides which agents to call based on natural language understanding.
    """
    
    def __init__(self):
        print("🚀 Initializing Coordinator Agent with IBM watsonx.ai...")
        
        # Initialize specialist agents
        self.sentiment_agent = SentimentAgent()
        self.creative_agent = CreativeAgent()
        self.predictive_agent = PredictiveAgent()
        
        # Initialize watsonx.ai
        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        self.model = ModelInference(
            model_id=COORDINATOR_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params=COORDINATOR_PARAMETERS
        )
        
        # Define tools and conversation history
        self.tools = self._load_tools()
        self.conversation_history = []
        
        print(f"✅ Connected to watsonx.ai - Model: {COORDINATOR_ID}")
    
    def _load_tools(self) -> List[Dict[str, Any]]:
        """Load tool definitions from JSON file"""
        tools_file = Path(__file__).parent / "../data/descriptions/agent_descriptions.json"
        with open(tools_file, 'r') as f:
            config = json.load(f)
        return config["tools"]
    
    def process_director_request(self, request: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Main entry point for director requests.
        Uses watsonx.ai to intelligently determine which agents to call.
        """
        print(f"\n{'='*80}")
        print(f"🤖 Coordinator Agent: Processing request")
        print(f"{'='*80}")
        print(f"📝 Request: '{request}'")
        print(f"{'='*80}\n")
        
        workflow_result = {
            "request": request,
            "tool_calls": [],
            "agent_results": {},
            "final_response": "",
            "iterations": 0
        }
        
        self.conversation_history.append({"role": "user", "content": request})
        
        # iterative tool calling loop
        for iteration in range(1, max_iterations + 1):
            workflow_result["iterations"] = iteration
            print(f"\n🔄 Iteration {iteration}")
            
            # build prompt and call watsonx.ai
            prompt = self._build_prompt()
            print("🧠 Calling watsonx.ai for decision...")
            response = self.model.generate_text(prompt=prompt).strip()
            
            # parse response for tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No more tools to call - we have the final answer
                workflow_result["final_response"] = response
                self.conversation_history.append({"role": "assistant", "content": response})
                print("\n✅ Coordinator Agent: Workflow completed")
                break
            
            # execute all requested tools
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                parameters = tool_call["parameters"]
                
                print(f"🔧 Executing tool: {tool_name}")
                print(f"📋 Parameters: {json.dumps(parameters, indent=2)}")
                
                # Track and execute
                workflow_result["tool_calls"].append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "parameters": parameters
                })
                
                result = self._execute_tool(tool_name, parameters)
                
                if tool_name not in workflow_result["agent_results"]:
                    workflow_result["agent_results"][tool_name] = []
                workflow_result["agent_results"][tool_name].append(result)
                
                tool_results.append({"tool": tool_name, "result": result})
            
            # Add tool results to conversation
            self.conversation_history.append({
                "role": "tool",
                "content": json.dumps(tool_results, indent=2)
            })
        
        # Fallback if max iterations reached
        if not workflow_result["final_response"]:
            workflow_result["final_response"] = (
                "I've gathered information from the specialist agents. "
                "Please see the detailed results above."
            )
        
        print(f"\n{'='*80}")
        print("✅ Workflow Complete")
        print(f"{'='*80}\n")
        
        return workflow_result
    
    def _build_prompt(self) -> str:
        """Create prompt with system instructions, tools, and conversation history"""
        # Build conversation context
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.conversation_history
        ])
        
        # Build tool descriptions
        tools_text = "\n\nAvailable Tools:\n"
        for tool in self.tools:
            func = tool["function"]
            tools_text += f"\n- {func['name']}: {func['description']}\n"
            tools_text += f"  Parameters: {json.dumps(func['parameters'], indent=2)}\n"
        
        return f"""You are an intelligent NBA fan engagement coordinator.
        Your job is to understand user requests and determine which specialist agents to call.

        {tools_text}

        Conversation History:
        {conversation_text}

        Instructions:
        If you need to call a tool, respond with a JSON object in this format:
        {{"tool_calls": [{{"name": "tool_name", "parameters": {{"param": "value"}}}}]}}

        If you have all the information needed to answer, provide a natural language response without any JSON.
        Be concise but comprehensive in your final answer.

        Your response:"""
            
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model's response to extract tool calls"""
        try:
            if "tool_calls" in response and "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end != 0:
                    parsed = json.loads(response[start:end])
                    if "tool_calls" in parsed:
                        return parsed["tool_calls"]
            return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Error parsing tool calls: {e}")
            return []
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified specialist agent with given parameters"""
        try:
            if tool_name == "analyze_sentiment":
                result = self.sentiment_agent.analyze_sentiment(parameters)
                return {"status": "success", "data": result}
            
            elif tool_name == "generate_content":
                # Map to creative agent format
                params = {
                    "type": parameters.get("content_type", "hooks_and_events"),
                    "context": parameters.get("context", "recent_performance")
                }
                result = self.creative_agent.generate_content(params)
                return {"status": "success", "data": result}
            
            elif tool_name == "forecast_trends":
                result = self.predictive_agent.forecast_trends(parameters)
                return {"status": "success", "data": result}
            
            else:
                return {"status": "error", "message": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            print(f"❌ Error executing tool {tool_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    def reset_conversation(self):
        """Reset conversation history for a new session"""
        self.conversation_history = []
        print("🔄 Conversation history reset")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status for monitoring"""
        return {
            "conversation_length": len(self.conversation_history),
            "agent_status": {
                "sentiment_agent": "ready",
                "creative_agent": "ready",
                "predictive_agent": "ready"
            },
            "model": COORDINATOR_ID,
            "tools_available": len(self.tools)
        }