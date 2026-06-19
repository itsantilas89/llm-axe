from __future__ import annotations


def build_qa_questions() -> list[dict]:
    """Generate the fixed QA set used by batch QA generation and consistency validation."""
    return [
        {
            "id": "interest_rate",
            "query": "Ποιο είναι το επιτόκιο του προγράμματος;",
            "fields": ["interest_rate"],
            "check_type": "substring",
            "description": "Interest rate consistency",
        },
        {
            "id": "duration",
            "query": "Ποια είναι η διάρκεια του δανείου;",
            "fields": ["loan_duration", "duration"],
            "check_type": "substring",
            "description": "Loan duration consistency",
        },
        {
            "id": "eligible_parties",
            "query": "Ποιες κατηγορίες αιτούντων είναι επιλέξιμες;",
            "fields": ["eligible_parties"],
            "check_type": "list_contains",
            "description": "Eligible parties consistency",
        },
        {
            "id": "funding_amount",
            "query": "Ποιο είναι το ελάχιστο και μέγιστο ποσό χρηματοδότησης;",
            "fields": ["minimum_funding_amount", "maximum_funding_amount"],
            "check_type": "numeric_range",
            "description": "Funding amount consistency",
        },
        {
            "id": "eligibility_criteria",
            "query": "Ποια είναι τα κριτήρια επιλεξιμότητας;",
            "fields": ["eligibility_criteria"],
            "check_type": "list_contains",
            "description": "Eligibility criteria consistency",
        },
        {
            "id": "programme_name",
            "query": "Πώς ονομάζεται το πρόγραμμα;",
            "fields": ["programme_name"],
            "check_type": "substring",
            "description": "Programme name consistency",
        },
        {
            "id": "energy_interventions",
            "query": "Ποιες ενεργειακές παρεμβάσεις είναι επιλέξιμες;",
            "fields": ["eligible_interventions"],
            "check_type": "list_contains",
            "description": "Energy interventions consistency",
        },
        {
            "id": "funding_coverage",
            "query": "Ποιο είναι το ποσοστό κάλυψης του δανείου;",
            "fields": ["funding_coverage"],
            "check_type": "substring",
            "description": "Funding coverage consistency",
        },
        {
            "id": "completion_delay_consequences",
            "query": "Τι γίνεται αν δεν ολοκληρωθούν οι εργασίες μέσα στην προβλεπόμενη προθεσμία;",
            "fields": ["completion_delay_consequences"],
            "check_type": "substring",
            "description": "Completion delay consequences consistency",
        },
        {
            "id": "post_completion_obligations",
            "query": "Υπάρχουν υποχρεώσεις του δικαιούχου μετά την ολοκλήρωση του έργου;",
            "fields": ["post_completion_obligations"],
            "check_type": "substring",
            "description": "Post-completion obligations consistency",
        },
    ]
