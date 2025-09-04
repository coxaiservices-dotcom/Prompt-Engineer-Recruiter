"""
Interactive Prompt Pattern Examples
Demonstrates all 16 prompt engineering patterns with hands-on examples
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from demo_lib import get_llm, ChatSession

@dataclass
class PatternExample:
    name: str
    description: str
    template: str
    example_inputs: Dict[str, Any]
    expected_behavior: str
    use_cases: List[str]
    variations: List[str]

class InteractivePatternDemo:
    def __init__(self):
        self.llm = get_llm()
        self.patterns = self._load_pattern_examples()
    
    def _load_pattern_examples(self) -> Dict[str, PatternExample]:
        """Load all 16 pattern examples with detailed demonstrations"""
        
        patterns = {
            "persona": PatternExample(
                name="Persona Pattern",
                description="AI adopts a specific role/character with domain expertise",
                template="Act as a {role}. Task: {task}. Audience: {audience}. Domain: {domain}. Constraints: {constraints}.",
                example_inputs={
                    "role": "Senior Software Architect",
                    "task": "Review this system design for scalability issues",
                    "audience": "development team",
                    "domain": "e-commerce platform",
                    "constraints": "must handle 10x traffic growth"
                },
                expected_behavior="AI responds with expertise-level analysis, uses domain-specific terminology, considers audience knowledge level",
                use_cases=[
                    "Technical code reviews",
                    "Business strategy sessions", 
                    "Educational content creation",
                    "Customer service scenarios"
                ],
                variations=[
                    "Expert vs Beginner personas",
                    "Industry-specific experts",
                    "Cultural/regional adaptations",
                    "Multiple persona collaboration"
                ]
            ),
            
            "question_refinement": PatternExample(
                name="Question Refinement Pattern",
                description="AI clarifies ambiguous questions before answering",
                template="Original: {initial_question}. Goal: {goal}. Rewrite the question to be clearer. Then answer.",
                example_inputs={
                    "initial_question": "How do I make it better?",
                    "goal": "improve code performance"
                },
                expected_behavior="AI identifies ambiguity, asks clarifying questions, provides refined question, then answers",
                use_cases=[
                    "Requirement gathering",
                    "Problem definition",
                    "Research question formulation",
                    "Debugging assistance"
                ],
                variations=[
                    "Iterative refinement",
                    "Multiple interpretation handling",
                    "Context-aware refinement",
                    "Domain-specific clarification"
                ]
            ),
            
            "cognitive_verifier": PatternExample(
                name="Cognitive Verifier Pattern", 
                description="AI provides solution with self-verification and risk analysis",
                template="Problem: {problem}. Provide solution. Show reasoning. Verify against {acceptance_criteria}. List risks.",
                example_inputs={
                    "problem": "Database queries are running slowly",
                    "acceptance_criteria": "queries under 100ms, no data loss"
                },
                expected_behavior="AI gives solution, explains reasoning step-by-step, verifies solution meets criteria, identifies potential risks",
                use_cases=[
                    "Critical system decisions",
                    "Code review processes",
                    "Business strategy validation",
                    "Safety-critical systems"
                ],
                variations=[
                    "Multi-step verification",
                    "Peer review simulation",
                    "Risk mitigation planning",
                    "Confidence scoring"
                ]
            ),
            
            "audience_persona": PatternExample(
                name="Audience Persona Pattern",
                description="AI adapts communication style for specific audience",
                template="Rewrite the content for {audience_type} with {tone} tone at {reading_level} level.",
                example_inputs={
                    "audience_type": "non-technical executives",
                    "tone": "professional but accessible",
                    "reading_level": "high school",
                    "content": "Database normalization reduces redundancy through structured table relationships"
                },
                expected_behavior="AI adapts vocabulary, examples, and structure to match audience needs and comprehension level",
                use_cases=[
                    "Documentation writing",
                    "Presentation preparation",
                    "Educational content",
                    "Customer communications"
                ],
                variations=[
                    "Technical to business translation",
                    "Age-appropriate adaptations",
                    "Cultural sensitivity adjustments",
                    "Expertise level scaling"
                ]
            ),
            
            "flipped_interaction": PatternExample(
                name="Flipped Interaction Pattern",
                description="AI asks clarifying questions to understand requirements better",
                template="Goal: {goal}. Ask me up to {max_questions} clarifying questions. Then propose a plan.",
                example_inputs={
                    "goal": "optimize our application performance",
                    "max_questions": "5"
                },
                expected_behavior="AI asks strategic questions about current performance, bottlenecks, constraints, then provides tailored plan",
                use_cases=[
                    "Requirements gathering",
                    "Consulting engagements",
                    "Problem diagnosis",
                    "Solution customization"
                ],
                variations=[
                    "Progressive questioning",
                    "Domain-specific inquiry",
                    "Priority-based questioning",
                    "Interactive discovery"
                ]
            ),
            
            "game_play": PatternExample(
                name="Game Play Pattern",
                description="AI simulates scenarios with rules, roles, and outcomes",
                template="Scenario: {scenario}. Roles: {roles}. Win: {win_conditions}. Simulate 3 rounds and show lessons.",
                example_inputs={
                    "scenario": "Incident response simulation",
                    "roles": "DevOps engineer, Security analyst, Product manager",
                    "win_conditions": "System restored under 30 minutes with root cause identified"
                },
                expected_behavior="AI plays multiple roles, simulates realistic scenarios, shows decision consequences, extracts learning points",
                use_cases=[
                    "Training simulations",
                    "Risk scenario planning", 
                    "Decision making practice",
                    "Team dynamics exploration"
                ],
                variations=[
                    "Multi-player scenarios",
                    "Progressive difficulty",
                    "Real-time adaptations",
                    "Outcome branching"
                ]
            ),
            
            "template": PatternExample(
                name="Template Pattern",
                description="AI responds in exact structured format",
                template="Respond ONLY with these sections: {sections}.",
                example_inputs={
                    "sections": "Executive Summary, Technical Analysis, Risk Assessment, Recommendations, Next Steps"
                },
                expected_behavior="AI strictly follows provided structure, ensures all sections present, maintains consistent formatting",
                use_cases=[
                    "Report generation",
                    "Documentation standards",
                    "API responses",
                    "Data formatting"
                ],
                variations=[
                    "JSON/XML structures",
                    "Nested hierarchies",
                    "Conditional sections",
                    "Dynamic templates"
                ]
            ),
            
            "meta_language": PatternExample(
                name="Meta Language Creation Pattern",
                description="AI creates custom command languages for specific workflows",
                template="Define a command language with commands {commands} and rules {rules}. Give 3 examples.",
                example_inputs={
                    "commands": "ANALYZE, SUMMARIZE, COMPARE, RECOMMEND",
                    "rules": "commands can be chained with |, parameters in [], output format specified with >>"
                },
                expected_behavior="AI creates grammar, syntax rules, provides clear examples, enables complex workflow automation",
                use_cases=[
                    "Workflow automation",
                    "Domain-specific languages",
                    "Command interfaces",
                    "Process standardization"
                ],
                variations=[
                    "Context-sensitive commands",
                    "Hierarchical structures",
                    "Error handling syntax",
                    "Extension mechanisms"
                ]
            ),
            
            "recipe": PatternExample(
                name="Recipe Pattern",
                description="AI provides step-by-step procedures with checkpoints",
                template="Objective: {objective}. Resources: {resources}. Provide step-by-step procedure with checkpoints.",
                example_inputs={
                    "objective": "Deploy a microservice to production",
                    "resources": "Kubernetes cluster, CI/CD pipeline, monitoring tools"
                },
                expected_behavior="AI creates detailed steps, includes verification points, considers dependencies, provides troubleshooting",
                use_cases=[
                    "Process documentation",
                    "Training materials",
                    "Troubleshooting guides",
                    "Standard operating procedures"
                ],
                variations=[
                    "Parallel step execution",
                    "Conditional branching",
                    "Recovery procedures",
                    "Quality checkpoints"
                ]
            ),
            
            "alternative_approaches": PatternExample(
                name="Alternative Approaches Pattern",
                description="AI provides multiple solution options with trade-off analysis",
                template="Task: {task}. Provide {num_options} approaches. For each: Method, Pros, Cons, Risks.",
                example_inputs={
                    "task": "implement user authentication system",
                    "num_options": "3"
                },
                expected_behavior="AI presents diverse solutions, analyzes trade-offs objectively, considers different constraints and priorities",
                use_cases=[
                    "Architecture decisions",
                    "Technology selection",
                    "Problem-solving sessions",
                    "Strategic planning"
                ],
                variations=[
                    "Cost-benefit analysis",
                    "Timeline considerations",
                    "Risk-adjusted options",
                    "Hybrid approaches"
                ]
            ),
            
            "ask_for_input": PatternExample(
                name="Ask for Input Pattern", 
                description="AI requests specific information before proceeding",
                template="Task: {task}. Ask {num_questions} clarifying questions before proceeding.",
                example_inputs={
                    "task": "design database schema",
                    "num_questions": "4"
                },
                expected_behavior="AI asks relevant, specific questions that impact solution design, waits for responses before continuing",
                use_cases=[
                    "Custom solution design",
                    "Requirements analysis",
                    "Personalization",
                    "Interactive consultations"
                ],
                variations=[
                    "Progressive questioning",
                    "Conditional follow-ups",
                    "Priority-based inquiry",
                    "Expert-guided discovery"
                ]
            ),
            
            "outline_expansion": PatternExample(
                name="Outline Expansion Pattern",
                description="AI creates hierarchical structure then expands each section",
                template="Topic: {topic}. Provide outline with depth {depth}. Expand each section.",
                example_inputs={
                    "topic": "microservices architecture best practices",
                    "depth": "3 levels"
                },
                expected_behavior="AI creates logical hierarchy, ensures comprehensive coverage, expands with appropriate detail level",
                use_cases=[
                    "Documentation creation",
                    "Course curriculum design",
                    "Research organization",
                    "Content planning"
                ],
                variations=[
                    "Audience-specific depth",
                    "Interactive expansion",
                    "Cross-referenced sections",
                    "Multi-format output"
                ]
            ),
            
            "menu_actions": PatternExample(
                name="Menu Actions Pattern",
                description="AI provides interactive menu of options",
                template="Provide a menu of options: {menu_items}. Wait for user choice before proceeding.",
                example_inputs={
                    "menu_items": "1) Analyze current performance, 2) Design optimization strategy, 3) Review existing solutions, 4) Create implementation plan"
                },
                expected_behavior="AI presents clear options, waits for selection, executes chosen action with appropriate detail",
                use_cases=[
                    "Interactive workflows",
                    "Guided tutorials",
                    "Decision trees",
                    "User-driven exploration"
                ],
                variations=[
                    "Conditional menus",
                    "Nested options",
                    "Progress tracking",
                    "Dynamic generation"
                ]
            ),
            
            "fact_check": PatternExample(
                name="Fact Check List Pattern",
                description="AI breaks claims into verifiable facts with confidence levels",
                template="Claim: {claim}. Break into atomic facts. Provide evidence, confidence (High/Med/Low).",
                example_inputs={
                    "claim": "Kubernetes reduces infrastructure costs by 40% while improving application reliability"
                },
                expected_behavior="AI identifies individual verifiable statements, provides supporting evidence, assigns confidence ratings",
                use_cases=[
                    "Information verification",
                    "Research validation",
                    "Content quality assurance",
                    "Decision support"
                ],
                variations=[
                    "Source attribution",
                    "Contradictory evidence",
                    "Confidence reasoning",
                    "Update mechanisms"
                ]
            ),
            
            "tail_generation": PatternExample(
                name="Tail Generation Pattern",
                description="AI continues incomplete content in specified style",
                template="Continue this text in style {style}, about {length} words: {incomplete_text}",
                example_inputs={
                    "style": "technical documentation",
                    "length": "150",
                    "incomplete_text": "The microservice architecture pattern enables..."
                },
                expected_behavior="AI maintains consistent style, appropriate length, logical continuation, preserves original intent",
                use_cases=[
                    "Content completion",
                    "Style consistency",
                    "Draft enhancement",
                    "Writer's block assistance"
                ],
                variations=[
                    "Multiple style options",
                    "Length flexibility",
                    "Tone adaptation", 
                    "Context preservation"
                ]
            ),
            
            "semantic_filter": PatternExample(
                name="Semantic Filter Pattern",
                description="AI filters content based on semantic criteria",
                template="Filter text according to {filters}. Return cleaned text and list of changes. Text: {text}",
                example_inputs={
                    "filters": "remove technical jargon, simplify complex sentences, ensure professional tone",
                    "text": "The heterogeneous distributed system architecture facilitates seamless horizontal scalability."
                },
                expected_behavior="AI applies semantic rules, explains changes made, preserves core meaning while meeting filter criteria",
                use_cases=[
                    "Content sanitization",
                    "Audience adaptation",
                    "Quality assurance",
                    "Compliance checking"
                ],
                variations=[
                    "Multi-criteria filtering",
                    "Severity levels",
                    "Change tracking",
                    "Approval workflows"
                ]
            )
        }
        
        return patterns
    
    def demonstrate_pattern(self, pattern_name: str) -> None:
        """Demonstrate a specific pattern interactively"""
        if pattern_name not in self.patterns:
            print(f"Pattern '{pattern_name}' not found. Available patterns: {list(self.patterns.keys())}")
            return
        
        pattern = self.patterns[pattern_name]
        
        print(f"\n🎯 {pattern.name}")
        print("=" * 50)
        print(f"Description: {pattern.description}")
        print(f"\nTemplate: {pattern.template}")
        print(f"\nExpected Behavior: {pattern.expected_behavior}")
        
        # Show example
        print(f"\n📝 Example:")
        formatted_prompt = pattern.template.format(**pattern.example_inputs)
        print(f"Prompt: {formatted_prompt}")
        
        response = self.llm.invoke(formatted_prompt)
        print(f"Response: {response}")
        
        # Show use cases
        print(f"\n💼 Use Cases:")
        for use_case in pattern.use_cases:
            print(f"  • {use_case}")
        
        # Show variations
        print(f"\n🔄 Variations:")
        for variation in pattern.variations:
            print(f"  • {variation}")
    
    def interactive_pattern_explorer(self) -> None:
        """Interactive exploration of patterns"""
        print("🔍 INTERACTIVE PATTERN EXPLORER")
        print("=" * 50)
        print("Available patterns:")
        
        for i, (key, pattern) in enumerate(self.patterns.items(), 1):
            print(f"{i:2d}. {pattern.name}")
        
        while True:
            try:
                choice = input(f"\nSelect pattern (1-{len(self.patterns)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    break
                
                pattern_index = int(choice) - 1
                pattern_keys = list(self.patterns.keys())
                if 0 <= pattern_index < len(pattern_keys):
                    self.demonstrate_pattern(pattern_keys[pattern_index])
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number or 'q' to quit.")
    
    def pattern_comparison(self, pattern_names: List[str]) -> None:
        """Compare multiple patterns side by side"""
        print(f"\n📊 PATTERN COMPARISON: {', '.join(pattern_names)}")
        print("=" * 70)
        
        comparison_data = []
        for name in pattern_names:
            if name in self.patterns:
                pattern = self.patterns[name]
                comparison_data.append({
                    "name": pattern.name,
                    "description": pattern.description,
                    "primary_use": pattern.use_cases[0] if pattern.use_cases else "General",
                    "complexity": len(pattern.template.split('{'))  # Simple complexity measure
                })
        
        if not comparison_data:
            print("No valid patterns found for comparison.")
            return
        
        # Display comparison table
        print(f"{'Pattern':<25} {'Primary Use':<20} {'Complexity':<10} Description")
        print("-" * 70)
        for data in comparison_data:
            print(f"{data['name']:<25} {data['primary_use']:<20} {data['complexity']:<10} {data['description'][:30]}...")
    
    def pattern_workflow_demo(self) -> None:
        """Demonstrate chaining multiple patterns in a workflow"""
        print("\n🔗 PATTERN WORKFLOW DEMONSTRATION")
        print("=" * 50)
        print("Scenario: Creating comprehensive technical documentation")
        
        # Step 1: Use Question Refinement to understand requirements
        print("\n1. Question Refinement Pattern:")
        refinement_prompt = self.patterns["question_refinement"].template.format(
            initial_question="We need documentation",
            goal="create user-friendly API documentation"
        )
        print(f"Prompt: {refinement_prompt}")
        refinement_response = self.llm.invoke(refinement_prompt)
        print(f"Response: {refinement_response[:200]}...")
        
        # Step 2: Use Outline Expansion to structure content
        print("\n2. Outline Expansion Pattern:")
        outline_prompt = self.patterns["outline_expansion"].template.format(
            topic="API documentation for REST endpoints",
            depth="3 levels"
        )
        print(f"Prompt: {outline_prompt}")
        outline_response = self.llm.invoke(outline_prompt)
        print(f"Response: {outline_response[:200]}...")
        
        # Step 3: Use Template Pattern for consistent format
        print("\n3. Template Pattern:")
        template_prompt = self.patterns["template"].template.format(
            sections="Introduction, Authentication, Endpoints, Examples, Error Codes, Support"
        )
        print(f"Prompt: {template_prompt}")
        template_response = self.llm.invoke(template_prompt)
        print(f"Response: {template_response[:200]}...")
        
        # Step 4: Use Audience Persona for final adaptation
        print("\n4. Audience Persona Pattern:")
        audience_prompt = self.patterns["audience_persona"].template.format(
            audience_type="junior developers",
            tone="friendly and encouraging",
            reading_level="intermediate"
        )
        print(f"Prompt: {audience_prompt}")
        
        print("\n✅ Workflow complete! Four patterns combined for comprehensive documentation.")
    
    def pattern_effectiveness_analysis(self) -> None:
        """Analyze when to use which patterns"""
        print("\n📈 PATTERN EFFECTIVENESS ANALYSIS")
        print("=" * 50)
        
        categories = {
            "Structure & Control": ["template", "recipe", "outline_expansion", "menu_actions"],
            "Safety & Accuracy": ["question_refinement", "cognitive_verifier", "fact_check", "ask_for_input"],
            "Creativity & Adaptability": ["persona", "audience_persona", "flipped_interaction", "game_play", "alternative_approaches"],
            "Processing & Filtering": ["meta_language", "tail_generation", "semantic_filter"]
        }
        
        for category, pattern_list in categories.items():
            print(f"\n🎯 {category}:")
            for pattern_key in pattern_list:
                if pattern_key in self.patterns:
                    pattern = self.patterns[pattern_key]
                    primary_use = pattern.use_cases[0] if pattern.use_cases else "General"
                    print(f"  • {pattern.name}: Best for {primary_use}")
        
        print(f"\n💡 Selection Guidelines:")
        print("  • Use Structure patterns for consistent outputs")
        print("  • Use Safety patterns for critical decisions") 
        print("  • Use Creativity patterns for exploration")
        print("  • Use Processing patterns for content transformation")

def run_pattern_demo():
    """Run the complete interactive pattern demonstration"""
    demo = InteractivePatternDemo()
    
    print("🎨 INTERACTIVE PROMPT PATTERN EXAMPLES")
    print("=" * 50)
    
    # Quick demonstration of key patterns
    key_patterns = ["persona", "cognitive_verifier", "template", "alternative_approaches"]
    
    print("🚀 Quick Pattern Showcase:")
    for pattern in key_patterns:
        demo.demonstrate_pattern(pattern)
    
    # Pattern comparison
    demo.pattern_comparison(["persona", "template", "cognitive_verifier"])
    
    # Workflow demonstration  
    demo.pattern_workflow_demo()
    
    # Effectiveness analysis
    demo.pattern_effectiveness_analysis()
    
    print("\n" + "=" * 50)
    print("✅ Pattern examples demonstrated!")
    print("\nNext steps:")
    print("• Try the interactive explorer")
    print("• Experiment with pattern combinations")
    print("• Create custom variations")
    print("• Build domain-specific pattern libraries")
    
    # Optional: Run interactive explorer
    user_input = input("\nWould you like to explore patterns interactively? (y/n): ")
    if user_input.lower() == 'y':
        demo.interactive_pattern_explorer()

if __name__ == "__main__":
    run_pattern_demo()
