"""
Predictive Agent - Solution 1: Enhanced Prompt Engineering
Uses improved prompts to handle multi-team queries intelligently
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
    Intelligent predictive agent with enhanced multi-team query handling.
    Uses improved prompt engineering to recognize and handle multiple teams.
    """
    
    MAX_ITERATIONS = 6  # Increased for multi-team queries
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
        
        print("✅ Predictive Agent initialized with enhanced multi-team support")
    
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
            # Serialize with default=str to handle numpy/pandas types
            try:
                tool_content = json.dumps(tool_results, indent=2, default=str)
            except (TypeError, ValueError) as e:
                print(f"⚠️ Serialization warning: {e}")
                tool_content = str(tool_results)

            self.conversation_history.append({
                "role": "tool",
                "content": tool_content
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
        """Build enhanced prompt for AI to decide next action with multi-team support"""
        tools_text = self._format_tool_catalog()
        conversation_text = self._format_conversation_history()
        
        return f"""You are an intelligent predictive analytics agent for NBA fan engagement and team performance forecasting.

Your job is to analyze requests about future trends, predictions, and forecasts, then determine which tool methods to call to gather the necessary data.

⚠️ CRITICAL: TOOL SELECTION RULES ⚠️

DATA_TOOL - Use for ALL questions about team/player PERFORMANCE and GAME OUTCOMES:
- Provides REAL-TIME NBA stats: current season records, recent game results, player stats
- Provides HISTORICAL performance: past seasons, win/loss trends, playoff history
- Available for ALL 30 NBA teams with team_name parameter
- Methods: get_all_nba_teams, get_teams_by_rank, get_performance_metrics, get_recent_games, calculate_momentum_score, get_standings, get_competitive_context, analyze_performance_trends, get_historical_performance
- ✅ USE DATA_TOOL FOR:
  * "Who will win the championship?"
  * "Lakers championship chances?"
  * "How are Lakers doing?"
  * "Who would win Lakers vs Celtics?"
  * "Best team in the league?"
  * "Top championship contenders?"
  * ANY question about wins, losses, performance, matchups, or championships

FORECAST_TOOL - Use ONLY for NON-PERFORMANCE metrics (attendance, valuations, social engagement):
- Provides ML FORECASTING for: attendance numbers, team valuations, player salaries, social media metrics
- Uses historical statistical datasets (NOT real-time game data)
- ❌ CANNOT predict game wins, team performance, or championship outcomes
- ❌ DO NOT use for performance questions
- ✅ USE FORECAST_TOOL FOR:
  * "Predict attendance for next game"
  * "Fan engagement trends"
  * "Social media growth forecasts"
  * "Team valuation predictions"

🚨 MANDATORY WORKFLOW FOR CHAMPIONSHIP/PERFORMANCE QUESTIONS:
1. Questions like "Who will win championship?" or "Best team?" → ONLY use data_tool methods
2. First call get_all_nba_teams() to see all teams
3. Then call get_standings() to identify top teams
4. Then call data_tool methods for top 5-10 teams:
   - get_performance_metrics(team_name="Team Name")
   - calculate_momentum_score(team_name="Team Name")
   - get_competitive_context(team_name="Team Name")
   - analyze_performance_trends(team_name="Team Name")
5. Compare the data and make prediction based on real stats
6. NEVER use forecast_tool for performance predictions!

CRITICAL: MULTI-TEAM QUERY HANDLING
When the user's question involves MULTIPLE TEAMS (comparisons, matchups, "vs", "who would win", rankings):
1. IDENTIFY ALL TEAM NAMES mentioned in the question
2. Call the SAME data_tool methods for EACH team SEPARATELY using the team_name parameter
3. You MUST gather data for ALL teams before providing analysis
4. NEVER use forecast_tool for team performance comparisons

EXAMPLES OF MULTI-TEAM QUERIES:
- "Who would win: Orlando Magic vs Boston Celtics?"
  → Call get_performance_metrics(team_name="Orlando Magic")
  → Call get_performance_metrics(team_name="Boston Celtics")
  → Call calculate_momentum_score(team_name="Orlando Magic")
  → Call calculate_momentum_score(team_name="Boston Celtics")
  → Then compare and provide prediction

- "Compare Lakers, Warriors, and Celtics performance"
  → Call data_tool methods for ALL THREE teams

- "How do the Magic and Celtics match up?"
  → Call data_tool methods for BOTH teams

CHAMPIONSHIP/BEST TEAM QUERIES (NO SPECIFIC TEAMS MENTIONED):
- "Who will win the championship?"
  → Step 1: Call get_teams_by_rank(start_rank=1, end_rank=6) to get top 6 teams with comprehensive data
  → Step 2: Compare performance metrics, momentum scores, and trends from returned data
  → Step 3: Predict winner based on data
  → ❌ DO NOT use forecast_tool!
  → ❌ DO NOT call get_all_nba_teams() then manually query each team (inefficient)

- "Best team in the league?"
  → Call get_teams_by_rank(start_rank=1, end_rank=1) to get #1 ranked team with full stats
  → OR get_teams_by_rank(start_rank=1, end_rank=3) to compare top 3

- "Top 10 teams?"
  → Call get_teams_by_rank(start_rank=1, end_rank=10)

🎯 EFFICIENT WORKFLOW: Use get_teams_by_rank() instead of calling individual methods for multiple teams

SINGLE-TEAM vs MULTI-TEAM DETECTION:
- Single team: "How are the Lakers doing?" → Use team_name="Los Angeles Lakers"
- Two teams: "Magic vs Celtics" → Use team_name for EACH team
- Multiple teams: "Compare top 5 teams" → Use team_name for ALL teams
- NO TEAM SPECIFIED: If question doesn't mention specific teams (e.g., "who's the best team?", "championship favorites", "top performers"):
  * First call get_all_nba_teams() to get list of teams
  * Then call get_standings() without team_name to understand league rankings
  * Based on standings, identify relevant teams (top 5-10) 
  * Call performance methods for those specific teams using their full names
  * Example workflow: get_all_nba_teams() → get_standings() → identify top teams → get_performance_metrics for each
- Default team: If no team mentioned AND question is Lakers-specific context, omit team_name parameter (defaults to Lakers)

{tools_text}

{conversation_text}

INSTRUCTIONS:
1. If you need to gather more data, respond with a JSON object listing tool calls:
   {{"tool_calls": [{{"method": "method_name", "parameters": {{"param": "value", "team_name": "Full Team Name"}}}}]}}

2. For MULTI-TEAM queries, make MULTIPLE tool calls in ONE response (you can call the same method multiple times with different team_name values)

3. For performance/championship questions, use efficient data gathering:
   - For questions about top teams/championship: call get_teams_by_rank(start_rank=1, end_rank=6)
   - For single team questions: call individual methods with team_name parameter
   - For multi-team comparisons: call get_teams_by_rank with appropriate range OR individual methods
   ❌ DO NOT use forecast_tool methods for these questions!

4. If you have gathered sufficient data to answer the question, provide a comprehensive natural language analysis

5. For MATCHUP PREDICTIONS specifically:
   - Get performance metrics for BOTH teams
   - Get momentum scores for BOTH teams
   - Get recent trends for BOTH teams
   - Compare win rates, momentum, recent form, scoring efficiency
   - Predict winner based on data with confidence percentage

6. TEAM NAME FORMAT: Always use full official team names from get_all_nba_teams():
   - "Los Angeles Lakers" (not "Lakers")
   - "Boston Celtics" (not "Celtics")
   - "Orlando Magic" (not "Magic")
   - "Golden State Warriors" (not "Warriors")
   - "Miami Heat" (not "Heat")
   - When unsure of team names, call get_all_nba_teams() first

7. HANDLING QUESTIONS WITHOUT SPECIFIC TEAMS:
   - "Who's the best team?" → get_all_nba_teams(), then get_standings(), analyze top 3-5
   - "Championship favorites?" → get_standings(), identify top 6 teams, get metrics for each
   - "Top performers?" → get_standings() to find top teams, then get_top_performers for those teams
   - "Best team right now?" → get_competitive_context() for top ranked teams
   - Always use get_all_nba_teams() when you need to identify which teams to analyze

8. When analyzing data, consider:
   - Current performance trends
   - Historical patterns
   - Statistical predictions
   - Momentum indicators
   - Win/loss records
   - Competitive tier (championship contender vs play-in vs rebuild)

8. Use specific numbers and metrics in your analysis - don't be vague

⚠️ CRITICAL: For ANY question about team performance, wins, or championships → Use data_tool methods FIRST
⚠️ NEVER say "we need more data" when data_tool methods exist for the question

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

IMPORTANT: If this is a MULTI-TEAM comparison or matchup prediction:
1. Compare the metrics across ALL teams
2. Identify which team has advantages in each area
3. Provide a clear prediction with confidence percentage
4. Explain the reasoning based on the data

Provide a natural language analysis that:
1. Answers the original question directly
2. Highlights key predictions and trends
3. Provides specific numbers and metrics
4. For matchups: states predicted winner with confidence %
5. Explains reasoning clearly
6. Use numbers for relevant metrics and insights

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