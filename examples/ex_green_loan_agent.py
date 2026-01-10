#!/usr/bin/env python
"""
Example: Using the GreenLoanAgent with user prompts
Demonstrates how the agent discovers, filters, and recommends products
based on user queries.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_axe.models import OllamaChat
from llm_axe.green_loan_agent import GreenLoanAgent


def main():
    """
    Example of using GreenLoanAgent with various user queries.
    """
    
    # Initialize LLM
    print("[INFO] Initializing OllamaChat...")
    try:
        llm = OllamaChat(model="llama3.2:latest")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        print("Make sure Ollama is running: ollama serve")
        return
    
    # Initialize GreenLoanAgent
    print("[INFO] Initializing GreenLoanAgent...\n")
    agent = GreenLoanAgent(llm, temperature=0.7)
    
    # Example 1: User query about solar installation
    print("=" * 70)
    print("EXAMPLE 1: User Query - Solar Installation Loan")
    print("=" * 70)
    
    user_query_1 = (
        "Θέλω να εγκαταστήσω φωτοβολταϊκά στο σπίτι μου. "
        "Έχει κάποια τράπεζα δάνειο για αυτό;"
    )
    
    print(f"\nUser: {user_query_1}\n")
    recommendation_1 = agent.recommend(user_query_1)
    print(f"Agent:\n{recommendation_1}")
    
    # Example 2: User query about building insulation
    print("\n" + "=" * 70)
    print("EXAMPLE 2: User Query - Building Insulation")
    print("=" * 70)
    
    user_query_2 = (
        "Το σπίτι μας χάνει πολλή θερμότητα το χειμώνα. "
        "Θέλουμε να βάλουμε καλή μόνωση και να αντικαταστήσουμε τα παράθυρα. "
        "Υπάρχει χρηματοδότηση για αυτά;"
    )
    
    print(f"\nUser: {user_query_2}\n")
    recommendation_2 = agent.recommend(user_query_2)
    print(f"Agent:\n{recommendation_2}")
    
    # Example 3: User query about energy efficiency renovation
    print("\n" + "=" * 70)
    print("EXAMPLE 3: User Query - Complete Energy Efficiency Renovation")
    print("=" * 70)
    
    user_query_3 = (
        "Θέλω να κάνω ολική ενεργειακή αναβάθμιση του σπιτιού μου. "
        "Ποιες είναι οι καλύτερες επιλογές χρηματοδότησης;"
    )
    
    print(f"\nUser: {user_query_3}\n")
    recommendation_3 = agent.recommend(user_query_3)
    print(f"Agent:\n{recommendation_3}")
    
    # Follow-up question
    print("\n" + "=" * 70)
    print("FOLLOW-UP: User asks about interest rates")
    print("=" * 70)
    
    followup = "Ποια είναι τα επιτόκια σε αυτά τα δάνεια;"
    print(f"\nUser: {followup}\n")
    followup_response = agent.ask_followup(followup)
    print(f"Agent:\n{followup_response}")
    
    # Example 4: Non-related query (should be rejected)
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Non-Related Query (should be rejected)")
    print("=" * 70)
    
    unrelated_query = "Θέλω δάνειο για ένα αυτοκίνητο"
    print(f"\nUser: {unrelated_query}\n")
    response = agent.recommend(unrelated_query)
    print(f"Agent:\n{response}")


if __name__ == "__main__":
    main()
