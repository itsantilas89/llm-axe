"""
GreenLoanAgent - Intelligent product recommendation based on user queries
Inherits from Agent and uses VA4 discovery with user-prompt filtering
"""

import json
import sys
import os
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_axe.agents import Agent
from llm_axe.core import AgentType, make_prompt
from llm_axe.va4_product_discoverer import (
    discover_products,
    load_trusted_sources,
    scrape_page,
    extract_data,
    TEMPLATE_DEFAULT
)


class GreenLoanAgent(Agent):
    """
    An agent specialized in finding and recommending green loan products
    based on user queries. It uses trusted bank sources and validates
    results against the user's specific needs.
    """
    
    def __init__(self, llm, temperature=0.7, stream=False, **llm_options):
        """
        Initialize the GreenLoanAgent.
        
        Args:
            llm: An LLM object with ask() method (OllamaChat, etc.)
            temperature: LLM temperature (default 0.7)
            stream: Whether to stream responses
            **llm_options: Additional options for the LLM
        """
        # Initialize parent Agent with custom system prompt
        custom_system_prompt = self._get_system_prompt()
        super().__init__(
            llm=llm,
            custom_system_prompt=custom_system_prompt,
            temperature=temperature,
            stream=stream,
            **llm_options
        )
        self.trusted_sources = load_trusted_sources()
        self.discovered_products = []
        self.filtered_products = []
    
    @staticmethod
    def _get_system_prompt() -> str:
        """Return the system prompt for the Green Loan Agent."""
        return """You are a specialized financial advisor for green loans and energy efficiency financing.
Your role is to:
1. Understand the user's energy efficiency goals
2. Recommend relevant loan products from trusted Greek banks
3. Provide clear, accurate information about loan terms, rates, and eligibility
4. Help users understand which interventions are covered (solar, insulation, windows, etc.)

Always be accurate and only reference information you have verified from official bank sources.
Do not hallucinate or invent loan products or terms.
If you're unsure about something, ask the user for clarification or admit uncertainty.

Focus on:
- Energy efficiency renovations
- Building upgrades
- Solar/renewable energy installations
- Sustainable home improvements
- Green loan products from major Greek banks (Eurobank, Piraeus, NBG, Alpha, Crediabank)"""
    
    def evaluate_user_prompt(self, user_query: str) -> Dict:
        """
        Evaluate if user query is related to green loans/energy efficiency.
        
        Args:
            user_query: User's question or request
            
        Returns:
            Dictionary with relevance score and extracted topics
        """
        system = (
            "You are an expert at understanding user needs related to green loans and energy efficiency. "
            "Analyze the user query and respond with JSON: "
            "{\"is_relevant\": true/false, \"confidence\": 0-1, \"topics\": [list of topics], "
            "\"interventions\": [list of interventions mentioned]}"
        )
        
        user = (
            f"User query: {user_query}\n\n"
            "Is this query related to green loans, energy efficiency, building renovations, or "
            "sustainable energy upgrades? Extract the main topics and interventions they're interested in. "
            "Respond with JSON only."
        )
        
        prompts = [make_prompt("system", system), make_prompt("user", user)]
        
        try:
            response = self.llm.ask(prompts, format="json", temperature=0.2)
            result = json.loads(response.strip())
            return result
        except Exception as e:
            print(f"[WARN] Evaluation failed: {e}")
            return {
                "is_relevant": False,
                "confidence": 0,
                "topics": [],
                "interventions": []
            }
    
    def discover_relevant_products(self, user_query: str) -> List[Dict]:
        """
        Discover and filter products relevant to user query.
        
        Args:
            user_query: User's question or request
            
        Returns:
            List of relevant products with verification data
        """
        print(f"\n🔍 Analyzing user query: {user_query[:50]}...")
        
        # Step 1: Evaluate relevance
        evaluation = self.evaluate_user_prompt(user_query)
        
        if not evaluation.get("is_relevant", False):
            print("❌ Query is not related to green loans/energy efficiency")
            return []
        
        print(f"✅ Query is relevant (confidence: {evaluation.get('confidence', 0):.1%})")
        print(f"   Topics: {', '.join(evaluation.get('topics', [])[:3])}")
        
        # Step 2: Discover products from all trusted sources
        print(f"\n📥 Discovering products from {len(self.trusted_sources.get('greek_banks_green_loans_stegastika', []))} banks...")
        
        all_products = []
        
        if "greek_banks_green_loans_stegastika" in self.trusted_sources:
            urls = self.trusted_sources["greek_banks_green_loans_stegastika"]
            bank_names = [
                "Eurobank",
                "Crediabank", 
                "Piraeus Bank",
                "National Bank of Greece",
                "Alpha Bank"
            ]
            
            for bank_name, bank_url in zip(bank_names, urls):
                try:
                    print(f"  → Processing {bank_name}...")
                    products = discover_products(bank_name, bank_url, self.llm)
                    
                    # Extract nested links from each product
                    for product in products:
                        nested_links = self._extract_nested_links_from_product(
                            product.get("source_url", ""),
                            product.get("description", "")
                        )
                        product["nested_links"] = nested_links
                    
                    all_products.extend(products)
                    print(f"     Found {len(products)} products")
                except Exception as e:
                    print(f"     [ERROR] {e}")
                    continue
        
        print(f"✅ Discovered {len(all_products)} total products")
        self.discovered_products = all_products
        
        # Step 3: Filter by relevance to user query
        print(f"\n🎯 Filtering products for relevance to user query...")
        filtered = self._filter_products_by_relevance(
            all_products,
            user_query,
            evaluation.get("interventions", [])
        )
        
        print(f"✅ Filtered to {len(filtered)} relevant products")
        self.filtered_products = filtered
        
        return filtered
    
    def _extract_nested_links_from_product(self, product_url: str, product_text: str) -> List[Dict]:
        """
        Extract additional product links from within a product page.
        
        Args:
            product_url: URL of the product page
            product_text: Text content of the product page
            
        Returns:
            List of nested product links found
        """
        nested_links = []
        
        try:
            response = requests.get(product_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find all internal links
            for link in soup.find_all("a", href=True):
                href = link.get("href", "").strip()
                if not href or href.startswith("#"):
                    continue
                
                # Convert to absolute URL
                full_url = urljoin(product_url, href)
                
                # Filter: same domain only
                if urlparse(full_url).netloc != urlparse(product_url).netloc:
                    continue
                
                # Skip generic links
                if any(skip in full_url.lower() for skip in ["page=", "search", "sort", "login"]):
                    continue
                
                link_text = link.get_text(strip=True)[:100]
                
                # Only include if it looks like a product link
                if any(keyword in link_text.lower() or keyword in full_url.lower() 
                       for keyword in ["loan", "δάνειο", "product", "προϊόν", "green", "πράσινο", "energy", "ενέργεια"]):
                    nested_links.append({
                        "url": full_url,
                        "text": link_text,
                        "source_product_url": product_url
                    })
        
        except Exception as e:
            print(f"[WARN] Failed to extract nested links from {product_url}: {e}")
        
        return nested_links[:5]  # Limit to 5 nested links
    
    def _filter_products_by_relevance(self, products: List[Dict], user_query: str, 
                                     interventions: List[str]) -> List[Dict]:
        """
        Filter products by relevance to user query.
        
        Args:
            products: List of discovered products
            user_query: User's query
            interventions: List of interventions from evaluation
            
        Returns:
            Filtered list of relevant products
        """
        system = (
            "You are an expert at matching users with relevant financial products. "
            "Given a user query and a list of products, rank them by relevance. "
            "Return JSON: {\"ranked_products\": [{\"index\": int, \"relevance_score\": 0-1, \"reason\": str}]}"
        )
        
        products_summary = json.dumps([
            {
                "index": i,
                "name": p.get("programme_name", "Unknown"),
                "bank": p.get("bank_name", "Unknown"),
                "description": p.get("description", "")[:200],
                "eligible_interventions": p.get("eligible_interventions", [])[:5]
            }
            for i, p in enumerate(products[:15])  # Max 15 products to rank
        ], ensure_ascii=False)
        
        user = (
            f"User query: {user_query}\n"
            f"Interested in: {', '.join(interventions)}\n\n"
            f"Products:\n{products_summary}\n\n"
            f"Rank these products by relevance to the user's needs. "
            f"Return JSON with relevance scores (0-1) and reasons."
        )
        
        prompts = [make_prompt("system", system), make_prompt("user", user)]
        
        try:
            response = self.llm.ask(prompts, format="json", temperature=0.3)
            ranking = json.loads(response.strip())
            ranked = ranking.get("ranked_products", [])
            
            # Sort products by relevance score
            sorted_products = sorted(
                [(p, next((r.get("relevance_score", 0) for r in ranked if r.get("index") == i), 0))
                 for i, p in enumerate(products)],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Filter: only keep products with relevance > 0.5
            filtered = [p for p, score in sorted_products if score > 0.5]
            
            # Attach relevance scores
            for i, p in enumerate(filtered):
                p["user_relevance_score"] = sorted_products[i][1]
            
            return filtered[:10]  # Top 10 most relevant
            
        except Exception as e:
            print(f"[WARN] Filtering failed: {e}")
            # Fallback: return all products sorted by verification confidence
            return sorted(products, 
                         key=lambda x: x.get("verification", {}).get("confidence_score", 0),
                         reverse=True)[:10]
    
    def recommend(self, user_query: str) -> str:
        """
        Main entry point: analyze user query and provide recommendations.
        
        Args:
            user_query: User's question or request
            
        Returns:
            Formatted recommendation response
        """
        print(f"\n{'='*70}")
        print(f"GreenLoanAgent Recommendation Engine")
        print(f"{'='*70}")
        
        # Discover and filter products
        relevant_products = self.discover_relevant_products(user_query)
        
        if not relevant_products:
            return (
                "Δυστυχώς, δεν βρήκα προϊόντα που να ταιριάζουν με την ερώτησή σας. "
                "Σας προτείνω να επικοινωνήσετε απευθείας με τις τράπεζες ή να "
                "προσδιορίσετε καλύτερα τις ανάγκες σας."
            )
        
        # Generate recommendation response
        system = (
            "You are a helpful financial advisor for green loans. "
            "Based on the discovered products and user query, provide personalized recommendations. "
            "Be clear, accurate, and helpful. Compare products when relevant. "
            "Highlight key terms, interest rates, and eligible interventions. "
            "Respond in Greek."
        )
        
        products_info = json.dumps([
            {
                "name": p.get("programme_name", "Unknown"),
                "bank": p.get("bank_name", "Unknown"),
                "description": p.get("description", "")[:300],
                "interest_rate": p.get("interest_rate", "N/A"),
                "loan_duration": p.get("loan_duration", "N/A"),
                "eligible_interventions": p.get("eligible_interventions", [])[:5],
                "relevance_score": p.get("user_relevance_score", 0),
                "confidence": p.get("verification", {}).get("confidence_score", 0),
                "source_url": p.get("source_url", "")
            }
            for p in relevant_products
        ], ensure_ascii=False, indent=2)
        
        user = (
            f"User Query: {user_query}\n\n"
            f"Recommended Products:\n{products_info}\n\n"
            f"Provide personalized recommendations based on these products and the user's query. "
            f"Compare the best options and explain why they might be suitable. "
            f"Include practical next steps for the user."
        )
        
        prompts = [make_prompt("system", system), make_prompt("user", user)]
        
        try:
            recommendation = self.llm.ask(prompts, temperature=self.temperature, stream=self.stream)
            return recommendation
        except Exception as e:
            print(f"[ERROR] Failed to generate recommendation: {e}")
            return "Σφάλμα κατά τη δημιουργία συστάσεων. Παρακαλώ δοκιμάστε ξανά."
    
    def ask_followup(self, followup_query: str) -> str:
        """
        Handle follow-up questions about recommended products.
        
        Args:
            followup_query: Follow-up question from user
            
        Returns:
            Response to follow-up question
        """
        if not self.filtered_products:
            return "Παρακαλώ κάντε πρώτα μια αρχική ερώτηση για σύστάσεις."
        
        products_context = json.dumps([
            {
                "name": p.get("programme_name", "Unknown"),
                "bank": p.get("bank_name", "Unknown"),
                "interest_rate": p.get("interest_rate", "N/A"),
                "eligible_interventions": p.get("eligible_interventions", [])[:3]
            }
            for p in self.filtered_products
        ], ensure_ascii=False)
        
        system = (
            "You are a financial advisor helping with green loan questions. "
            "Use the provided product information to answer follow-up questions accurately."
        )
        
        user = (
            f"Available products:\n{products_context}\n\n"
            f"Follow-up question: {followup_query}\n\n"
            f"Answer in Greek based on the product information provided."
        )
        
        prompts = [make_prompt("system", system), make_prompt("user", user)]
        
        try:
            response = self.llm.ask(prompts, temperature=self.temperature)
            return response
        except Exception as e:
            return f"Σφάλμα: {e}"
