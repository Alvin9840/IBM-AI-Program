"""
Sentiment Agent - Historical Fan Sentiment Analysis Specialist
Uses IBM watsonx.ai to intelligently analyze historical Twitter sentiment patterns
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import is_dataclass, asdict

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

from tools.sentiment_tool import SentimentTool

from config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    SENTIMENT_ID,
    SENTIMENT_PARAMETERS
)


class SentimentAgent:
    """
    Analyzes historical NBA Twitter data to understand past fan sentiment patterns.
    Uses watsonx.ai to dynamically determine which sentiment_tool methods to call.
    """
    
    MAX_ITERATIONS = 5
    TOOL_RESULT_PREVIEW_LENGTH = 500
    DATA_PREVIEW_LENGTH = 2000
    
    FOCUS_QUESTION_MAP = {
        "recent_game": "What did fans historically say about game outcomes and team performance?",
        "player_performance": "What sentiment patterns exist in historical data about player performance?",
        "team_overall": "What are the historical fan sentiment patterns about the team overall?",
        "platform_breakdown": "How does historical fan sentiment vary across different social media platforms?",
        "trending_topics": "What topics historically trended among NBA fans and what sentiment did they have?"
    }
    
    def __init__(self):
        print("🔍 Initializing Sentiment Agent with IBM watsonx.ai...")
        
        self.sentiment_tool = SentimentTool()
        
        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        self.model = ModelInference(
            model_id=SENTIMENT_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params=SENTIMENT_PARAMETERS
        )
        
        self.tool_catalog = self._load_tool_catalog()
        self.conversation_history = []
        
        print("✅ Sentiment Agent initialized with dynamic tool selection")
    
    def _load_tool_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load tool method definitions from JSON files."""
        catalog_path = Path(__file__).parent / "../data/descriptions/sentiment_tool_descriptions.json"
        
        if not catalog_path.exists():
            raise FileNotFoundError(f"Tool catalog not found: {catalog_path}")
        
        with open(catalog_path, 'r') as f:
            return json.load(f)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert dataclasses and other non-serializable objects to dictionaries."""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        return obj
    
    def analyze_sentiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze historical Twitter sentiment patterns.
        
        Args:
            params: Dictionary with keys:
                - focus: Focus area ('recent_game', 'player_performance', etc.)
                - timeframe: Context timeframe
                - context: Optional current situation details for pattern matching
                - depth: Analysis depth
        
        Returns:
            Historical sentiment analysis with patterns from similar situations
        """
        print("🔍 Sentiment Agent: Analyzing historical sentiment patterns...")
        
        question = self._construct_question(params)
        print(f"📊 Analysis request: {question}")
        
        self.conversation_history = []
        result = self._execute_workflow(question, params)
        
        print("✅ Sentiment Agent: Historical analysis completed")
        return result
    
    def _construct_question(self, params: Dict[str, Any]) -> str:
        """Construct question that incorporates current context if provided."""
        focus = params.get("focus", "team_overall")
        context = params.get("context", {})
        
        if context:
            context_str = ", ".join(f"{k}: {v}" for k, v in context.items())
            return f"Given context ({context_str}), what did fans historically say in similar situations? Focus on {focus}."
        
        return self.FOCUS_QUESTION_MAP.get(focus, f"Analyze historical fan sentiment patterns related to {focus}")
    
    def _execute_workflow(self, question: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute iterative workflow where AI determines which tools to call."""
        workflow_result = {
            "question": question,
            "focus": params.get("focus", "team_overall"),
            "context_provided": bool(params.get("context")),
            "tool_calls": [],
            "data_gathered": {},
            "final_analysis": "",
            "iterations": 0
        }
        
        self.conversation_history.append({"role": "user", "content": question})
        
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            workflow_result["iterations"] = iteration
            print(f"\n🔄 Iteration {iteration}")
            
            prompt = self._build_decision_prompt()
            print("🧠 Consulting watsonx.ai for next action...")
            response = self.model.generate_text(prompt=prompt).strip()
            
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                workflow_result["final_analysis"] = response
                self.conversation_history.append({"role": "assistant", "content": response})
                print("✅ Analysis complete")
                break
            
            tool_results = self._execute_tools(tool_calls, workflow_result, iteration)
            self.conversation_history.append({
                "role": "tool",
                "content": json.dumps(tool_results, indent=2, default=str)
            })
        
        if not workflow_result["final_analysis"]:
            workflow_result["final_analysis"] = self._synthesize_final_analysis(
                question, workflow_result["data_gathered"]
            )
        
        return workflow_result
    
    def _execute_tools(self, tool_calls: List[Dict[str, Any]], 
                      workflow_result: Dict[str, Any], iteration: int) -> List[Dict[str, Any]]:
        """Execute all requested tool calls and track results."""
        tool_results = []
        
        for tool_call in tool_calls:
            method_name = tool_call["method"]
            parameters = tool_call.get("parameters", {})
            
            print(f"🔧 Executing: {method_name}")
            print(f"📋 Parameters: {json.dumps(parameters, indent=2)}")
            
            workflow_result["tool_calls"].append({
                "iteration": iteration,
                "method": method_name,
                "parameters": parameters
            })
            
            result = self._execute_tool_method(method_name, parameters)
            serializable_result = self._make_serializable(result)
            
            workflow_result["data_gathered"].setdefault(method_name, []).append(serializable_result)
            tool_results.append({"method": method_name, "result": serializable_result})
        
        return tool_results
    
    def _build_decision_prompt(self) -> str:
        """Build prompt for AI to decide next action."""
        tools_desc = "\n".join([
            f"{name}: {info['description']}\n  Params: {json.dumps(info.get('parameters', {}))}\n  Returns: {info['returns']}"
            for name, info in self.tool_catalog.items()
        ])
        
        history_preview = "\n".join([
            f"{msg['role'].upper()}: {msg['content'][:self.TOOL_RESULT_PREVIEW_LENGTH]}{'...' if len(msg['content']) > self.TOOL_RESULT_PREVIEW_LENGTH else ''}"
            for msg in self.conversation_history
        ])
        
        return f"""You are an intelligent sentiment analysis agent for NBA historical Twitter data.

CRITICAL: You analyze HISTORICAL sentiment data from past NBA Twitter posts, NOT real-time data.

AVAILABLE TOOLS:
{tools_desc}

WORKFLOW GUIDE:
1. For context-based questions: Use search_by_keyword to find similar historical situations
2. For overall patterns: Use get_overall_sentiment + get_sentiment_trends
3. For player focus: Use analyze_player_sentiment
4. For trending analysis: Use get_top_entities
5. For platform insights: Use get_platform_breakdown

CONVERSATION HISTORY:
{history_preview}

INSTRUCTIONS:
- To gather more data, respond with JSON: {{"tool_calls": [{{"method": "name", "parameters": {{}}}}]}}
- Can call multiple tools in one response
- When sufficient data gathered, provide analysis framed as historical patterns:
  * "Historically, fans said..."
  * "In similar past situations..."
  * Include specific percentages and metrics
  * Connect historical patterns to current context (if provided)

Your response:"""
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI response for tool calls."""
        if "tool_calls" not in response or "{" not in response:
            return []
        
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            if start == -1 or end == 0:
                return []
            
            return json.loads(response[start:end]).get("tool_calls", [])
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing tool calls: {e}")
            return []
    
    def _execute_tool_method(self, method_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool method with error handling."""
        if method_name not in self.tool_catalog:
            return {"error": f"Unknown method: {method_name}"}
        
        try:
            method = getattr(self.sentiment_tool, method_name, None)
            if not method:
                return {"error": f"Method {method_name} not found"}
            
            return {"status": "success", "data": method(**parameters)}
            
        except Exception as e:
            print(f"❌ Error executing {method_name}: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def _synthesize_final_analysis(self, question: str, data_gathered: Dict[str, Any]) -> str:
        """Synthesize final analysis if AI didn't provide one."""
        data_preview = json.dumps(data_gathered, indent=2, default=str)[:self.DATA_PREVIEW_LENGTH]
        
        prompt = f"""Based on HISTORICAL sentiment data, provide comprehensive analysis.

ORIGINAL QUESTION: {question}

HISTORICAL DATA: {data_preview}

Provide structured analysis:
1. HISTORICAL CONTEXT - Overall patterns from past data
2. SENTIMENT PATTERNS - Percentages, polarity scores from historical tweets
3. KEY THEMES - Common topics/keywords from past discussions
4. ENGAGEMENT PATTERNS - Platform activity, retweet patterns
5. CURRENT CONTEXT CONNECTION (if applicable) - How patterns relate to current situation
6. INSIGHTS - Notable patterns and what they suggest

Use past tense: "Historically...", "Past fans said...", "Historical data shows..."
Include specific numbers and percentages.

Your analysis:"""
        
        return self.model.generate_text(prompt=prompt).strip()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "available_tools": len(self.tool_catalog),
            "conversation_length": len(self.conversation_history),
            "sentiment_tool_datasets": len(self.sentiment_tool.datasets) if hasattr(self.sentiment_tool, 'datasets') else 0
        }
    
    def reset_conversation(self):
        """Reset conversation history for a new session."""
        self.conversation_history = []
        print("🔄 Conversation history reset")