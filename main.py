"""
NBA Fan Engagement AI Tool
Main entry point for the agentic AI system
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from agents.coordinator_agent import CoordinatorAgent

def main():
    """Main application entry point"""
    print("🏀 NBA Fan Engagement AI Tool")
    print("=" * 50)
    print("Welcome, Senior NBA Marketing Director!")
    print("Your AI assistant is ready to help with fan engagement planning.")
    print()
    
    # Initialize the coordinator agent
    coordinator = CoordinatorAgent()
    
    # Example interactions
    example_queries = [
        "How are the Lakers performing right now?",
        "Whos likely to win the next NBA championship?",
        "How do the Warrior match up against the top teams in the West?",
        "What would fans say if the Orlando Magic won the NBA cup?",
        "Show me historical sentiment patterns for comeback wins",
        "Predict fan attendance for the next Wizards home game"
    ]
    
    print("💡 Example queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {query}")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("🎯 What would you like to know? (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the NBA Fan Engagement AI Tool!")
                break
            
            if not user_input:
                continue
            
            # Process the request
            print(f"\n🔄 Processing: '{user_input}'")
            print("-" * 60)
            
            result = coordinator.process_director_request(user_input)
            
            # Display the response
            print(result["final_response"])
            print("-" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again or contact support.")
            print()

def demonstrate_workflow():
    """Demonstrate the complete workflow with sample data"""
    print("🚀 Demonstrating NBA Fan Engagement AI Workflow")
    print("=" * 60)
    
    coordinator = CoordinatorAgent()
    
    # Sample comprehensive request
    sample_request = "Summarize what fans are saying about our last game, suggest content hooks, and forecast future trends"
    
    print(f"Sample Request: {sample_request}")
    print("-" * 60)
    
    result = coordinator.process_director_request(sample_request)
    
    print("FINAL RESPONSE:")
    print("=" * 60)
    print(result["final_response"])
    print("=" * 60)
    
    # Show workflow details
    print("\nWORKFLOW DETAILS:")
    print(f"Tasks executed: {', '.join(result['tasks'])}")
    print(f"Results generated: {list(result['results'].keys())}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demonstrate_workflow()
    else:
        main()