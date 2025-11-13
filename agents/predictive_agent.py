"""
Predictive Agent - Dynamic Trend Forecasting Specialist
Uses IBM watsonx.ai to intelligently determine which tools and methods to use
"""

import json
from typing import Dict, List, Any
from pathlib import Path
from dataclasses import is_dataclass, asdict

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

from tools.forecast_tool import ForecastTool
from tools.data_tool import DataTool

from config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    PREDICTIVE_ID,
    PREDICTIVE_PARAMETERS
)


class PredictiveAgent:
    """
    Intelligent predictive agent that uses watsonx.ai to dynamically determine
    which tool methods to call based on the prediction request.
    """
    
    MAX_ITERATIONS = 5
    TOOL_RESULT_PREVIEW_LENGTH = 500
    DATA_PREVIEW_LENGTH = 2000
    
    def __init__(self):
        print("🔮 Initializing Predictive Agent with IBM watsonx.ai...")
        
        self.forecast_tool = ForecastTool()
        self.data_tool = DataTool()
        
        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        self.model = ModelInference(
            model_id=PREDICTIVE_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params=PREDICTIVE_PARAMETERS
        )
        
        self.tool_catalog = self._load_tool_catalog()
        self.conversation_history = []
        
        print("✅ Predictive Agent initialized with dynamic tool selection")
    
    def _load_tool_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load tool method definitions from JSON files"""
        base_path = Path(__file__).parent / "../data/descriptions"
        tool_files = {
            "forecast_tool_descriptions.json": "forecast_tool",
            "data_tool_descriptions.json": "data_tool"
        }
        
        catalog = {}
        for filename, tool_type in tool_files.items():
            file_path = base_path / filename
            if not file_path.exists():
                print(f"⚠️ Warning: {file_path} not found")
                continue
                
            with open(file_path, 'r') as f:
                catalog.update(json.load(f))
        
        if not catalog:
            raise ValueError("No tool methods loaded. Ensure JSON files exist in data/")
        
        return catalog
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert dataclasses and other non-serializable objects to dictionaries"""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def forecast_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for predictive analysis using watsonx.ai for intelligent tool selection.
        
        Args:
            params: Dictionary with optional keys:
                - question: Specific question to answer
                - timeframe: Time horizon for predictions
                - scope: Analysis scope
        
        Returns:
            Comprehensive prediction results with analysis
        """
        print("🔮 Predictive Agent: Starting intelligent trend analysis...")
        
        question = params.get("question") or self._construct_question(params)
        print(f"📊 Analysis request: {question}")
        
        self.conversation_history = []
        result = self._execute_workflow(question, params)
        
        print("✅ Predictive Agent: Analysis completed")
        return result
    
    def _construct_question(self, params: Dict[str, Any]) -> str:
        """Construct question from parameters when not explicitly provided"""
        timeframe = params.get("timeframe", "next_month")
        scope = params.get("scope", "comprehensive")
        return f"Forecast trends for timeframe: {timeframe}, scope: {scope}"
    
    def _execute_workflow(self, question: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute iterative workflow where AI determines which tools to call"""
        workflow_result = {
            "question": question,
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
        """Execute all requested tool calls and track results"""
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
        """Build prompt for AI to decide next action"""
        tools_text = self._format_tool_catalog()
        conversation_text = self._format_conversation_history()
        
        return f"""You are an intelligent predictive analytics agent for NBA fan engagement and team performance forecasting.

Your job is to analyze requests about future trends, predictions, and forecasts, then determine which tool methods to call to gather the necessary data.

IMPORTANT CONTEXT:
- For questions about GAME WINS, TEAM PERFORMANCE, or WIN PREDICTIONS: Use data_tool methods to get current performance, trends, and momentum
- For questions about SOCIAL METRICS or FAN ENGAGEMENT PREDICTIONS: Use forecast_tool with available datasets
- ALWAYS call list_available_metrics FIRST if you need to use forecast_tool datasets to see what's actually available
- data_tool provides real-time Lakers performance data (recent games, win streaks, standings, momentum)
- forecast_tool provides historical statistical datasets for correlation and forecasting

{tools_text}

{conversation_text}

INSTRUCTIONS:
1. If you need to gather more data, respond with a JSON object listing tool calls:
   {{"tool_calls": [{{"method": "method_name", "parameters": {{"param": "value", "team_name" : "value"}}}}]}}

2. You can call multiple tools in one response if needed.

3. If you have gathered sufficient data to answer the question, provide a comprehensive natural language analysis.

4. For GAME WIN PREDICTIONS specifically:
   - Use data_tool methods: get_recent_games, get_performance_metrics, analyze_performance_trends, calculate_momentum_score
   - Analyze win rate, trends, momentum, and recent performance
   - Make predictions based on current form and historical patterns

5. When analyzing data, consider:
   - Current performance trends
   - Historical patterns
   - Statistical predictions
   - Correlation insights
   - Momentum indicators

Your response:"""
    
    def _format_tool_catalog(self) -> str:
        """Format tool catalog for prompt"""
        tools_text = "\n\nAVAILABLE TOOLS:\n"
        for method_name, info in self.tool_catalog.items():
            tools_text += f"\n{method_name}:\n"
            tools_text += f"  Description: {info['description']}\n"
            if info['parameters']:
                tools_text += f"  Parameters: {json.dumps(info['parameters'], indent=4)}\n"
            tools_text += f"  Returns: {info['returns']}\n"
        return tools_text
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for prompt with truncation"""
        conversation_text = "\n\nCONVERSATION HISTORY:\n"
        for msg in self.conversation_history:
            role = msg['role'].upper()
            content = msg['content']
            if role == "TOOL" and len(content) > self.TOOL_RESULT_PREVIEW_LENGTH:
                content = content[:self.TOOL_RESULT_PREVIEW_LENGTH] + "..."
            conversation_text += f"\n{role}: {content}\n"
        return conversation_text
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI response for tool calls"""
        if "tool_calls" not in response or "{" not in response:
            return []
        
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                return []
            
            parsed = json.loads(response[start:end])
            return parsed.get("tool_calls", [])
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing tool calls: {e}")
            return []
    
    def _execute_tool_method(self, method_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool method with error handling"""
        if method_name not in self.tool_catalog:
            return {"error": f"Unknown method: {method_name}"}
        
        try:
            tool_info = self.tool_catalog[method_name]
            tool = self._get_tool(tool_info["tool"])
            
            if not tool:
                return {"error": f"Unknown tool: {tool_info['tool']}"}
            
            method = getattr(tool, method_name)
            result = method(**parameters)
            
            return {"status": "success", "data": result}
            
        except Exception as e:
            print(f"❌ Error executing {method_name}: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def _get_tool(self, tool_name: str):
        """Get tool instance by name"""
        tool_map = {
            "forecast_tool": self.forecast_tool,
            "data_tool": self.data_tool
        }
        return tool_map.get(tool_name)
    
    def _synthesize_final_analysis(self, question: str, data_gathered: Dict[str, Any]) -> str:
        """Synthesize final analysis if AI didn't provide one"""
        data_preview = json.dumps(data_gathered, indent=2, default=str)[:self.DATA_PREVIEW_LENGTH]
        
        prompt = f"""Based on the data gathered, provide a comprehensive predictive analysis.

ORIGINAL QUESTION: {question}

DATA GATHERED:
{data_preview}

Provide a natural language analysis that:
1. Answers the original question
2. Highlights key predictions and trends
3. Provides actionable insights
4. Includes relevant metrics and forecasts with numbers.

Your analysis:"""
        
        return self.model.generate_text(prompt=prompt).strip()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "available_tools": len(self.tool_catalog),
            "conversation_length": len(self.conversation_history),
            "forecast_tool_datasets": len(self.forecast_tool.datasets),
            "data_tool_cache_stats": self.data_tool.get_cache_stats()
        }