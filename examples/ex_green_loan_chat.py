#!/usr/bin/env python
"""
Interactive GreenLoanAgent Chat
Real-time conversation with the GreenLoanAgent for green loan recommendations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_axe.models import OllamaChat
from llm_axe.green_loan_agent import GreenLoanAgent


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 70)
    print("🌱 ΕΛΛΗΝΙΚΟΣ ΠΡΑΚΤΟΡΕΥΤΗΣ ΠΡΑΣΙΝΩΝ ΔΑΝΕΙΩΝ")
    print("   Greek Green Loan Agent - Interactive Chat")
    print("=" * 70)
    print("\n🤖 Καλώς ήρθατε! Είμαι ο Green Loan Agent.")
    print("   Μπορώ να σας βοηθήσω να βρείτε το κατάλληλο δάνειο")
    print("   για ενεργειακή αναβάθμιση του σπιτιού σας.\n")
    print("💡 Μπορώ να σας βοηθήσω με ερωτήσεις για:")
    print("   • Φωτοβολταϊκά συστήματα")
    print("   • Μόνωση κτιρίων")
    print("   • Αντικατάσταση παραθύρων")
    print("   • Ανανεώσιμες πηγές ενέργειας")
    print("   • Επιδοτήσεις και χρηματοδότηση\n")
    print("📝 Πληκτρολογήστε 'exit' για έξοδο")
    print("   Πληκτρολογήστε 'help' για βοήθεια\n")
    print("=" * 70 + "\n")


def print_help():
    """Print help message."""
    print("\n" + "-" * 70)
    print("📚 ΒΟΗΘΕΙΑ - Help Topics:")
    print("-" * 70)
    print("""
Δήλωση: Περιγράψτε τις ενεργειακές αναβαθμίσεις που θέλετε να κάνετε.

Παραδείγματα:
  ✓ "Θέλω φωτοβολταϊκά για το σπίτι μου"
  ✓ "Ψάχνω δάνειο για μόνωση και ανταλλαγή παραθύρων"
  ✓ "Πόσο κοστίζει η ενεργειακή αναβάθμιση;"
  ✓ "Ποια είναι τα επιτόκια;"

Ακολουθ-ερωτήσεις:
  ✓ "Ποια είναι τα προσόντα;"
  ✓ "Πόσο μπορώ να δανειστώ;"
  ✓ "Ποια η διάρκεια;"

Εντολές:
  • exit     - Έξοδος από το πρόγραμμα
  • help     - Εμφάνιση αυτού του μηνύματος
  • clear    - Εκκαθάριση ιστορικού
  • products - Εμφάνιση ανακαλυφθέντων προϊόντων
""")
    print("-" * 70 + "\n")


def print_products(agent):
    """Display discovered and filtered products."""
    if not agent.filtered_products:
        print("\n⚠️ Κανένα προϊόν δεν ανακαλύφθηκε ακόμα.")
        print("   Κάντε μια ερώτηση πρώτα για να ανακαλύψετε προϊόντα.\n")
        return
    
    print("\n" + "-" * 70)
    print(f"📦 ΑΝΑΚΑΛΥΦΘΕΝΤΑ ΠΡΟΪΟΝΤΑ ({len(agent.filtered_products)})")
    print("-" * 70)
    
    for i, product in enumerate(agent.filtered_products, 1):
        name = product.get("programme_name", "Unknown")
        bank = product.get("bank_name", "Unknown")
        rate = product.get("interest_rate", "N/A")
        duration = product.get("loan_duration", "N/A")
        relevance = product.get("user_relevance_score", 0)
        
        print(f"\n{i}. {name}")
        print(f"   Τράπεζα: {bank}")
        print(f"   Επιτόκιο: {rate}")
        print(f"   Διάρκεια: {duration}")
        print(f"   Σχετικότητα: {relevance:.1%}")
        print(f"   URL: {product.get('source_url', 'N/A')}")
    
    print("\n" + "-" * 70 + "\n")


def main():
    """Main interactive loop."""
    
    print_welcome()
    
    # Initialize LLM
    print("🔧 Αρχικοποίηση συστήματος...")
    try:
        llm = OllamaChat(model="deepseek-r1:latest")
        print("✅ LLM Έτοιμο\n")
    except Exception as e:
        print(f"❌ Σφάλμα αρχικοποίησης: {e}")
        print("   Βεβαιωθείτε ότι το Ollama τρέχει: ollama serve")
        return
    
    # Initialize agent
    print("🤖 Αρχικοποίηση Agent...")
    try:
        agent = GreenLoanAgent(llm, temperature=0.7)
        print("✅ Agent Έτοιμο\n")
    except Exception as e:
        print(f"❌ Σφάλμα: {e}")
        return
    
    # Main loop
    conversation_count = 0
    
    while True:
        try:
            user_input = input("👤 Εσείς: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Αντίο!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ("exit", "quit"):
            print("\n👋 Ευχαριστώ που χρησιμοποιήσατε τον Green Loan Agent!")
            break
        
        if user_input.lower() == "help":
            print_help()
            continue
        
        if user_input.lower() == "clear":
            agent.discovered_products = []
            agent.filtered_products = []
            print("\n🔄 Ιστορικό εκκαθαρίστηκε\n")
            continue
        
        if user_input.lower() == "products":
            print_products(agent)
            continue
        
        # Process user query
        conversation_count += 1
        
        if conversation_count == 1:
            # First query: run discovery
            print("\n🤖 Agent: Ανακαλύπτω σχετικά προϊόντα...\n")
            response = agent.recommend(user_input)
        else:
            # Follow-up query
            print("\n🤖 Agent: Απαντώ σε ακολουθούσα ερώτηση...\n")
            response = agent.ask_followup(user_input)
        
        print(f"🤖 Agent: {response}\n")


if __name__ == "__main__":
    main()
