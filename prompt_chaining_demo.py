"""
Advanced Prompt Chaining Workflows Demo
Demonstrates sequential, parallel, and conditional chaining patterns
"""

from demo_lib import get_llm, ChatSession
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from enum import Enum

class ChainType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel" 
    CONDITIONAL = "conditional"
    FEEDBACK_LOOP = "feedback_loop"

@dataclass
class ChainStep:
    name: str
    prompt_template: str
    inputs: List[str]
    outputs: List[str]
    depends_on: Optional[List[str]] = None
    condition: Optional[str] = None

@dataclass
class ChainResult:
    step_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    prompt_used: str
    response: str

class PromptChain:
    def __init__(self, llm, chain_type: ChainType = ChainType.SEQUENTIAL):
        self.llm = llm
        self.chain_type = chain_type
        self.steps: List[ChainStep] = []
        self.results: List[ChainResult] = []
        self.context: Dict[str, Any] = {}
    
    def add_step(self, step: ChainStep):
        """Add a step to the chain"""
        self.steps.append(step)
        return self
    
    def execute(self, initial_data: Dict[str, Any]) -> List[ChainResult]:
        """Execute the prompt chain with initial data"""
        self.context.update(initial_data)
        self.results = []
        
        if self.chain_type == ChainType.SEQUENTIAL:
            return self._execute_sequential()
        elif self.chain_type == ChainType.PARALLEL:
            return self._execute_parallel()
        elif self.chain_type == ChainType.CONDITIONAL:
            return self._execute_conditional()
        elif self.chain_type == ChainType.FEEDBACK_LOOP:
            return self._execute_feedback_loop()
        raise ValueError(f"Unsupported chain type: {self.chain_type}")
    
    def _execute_sequential(self) -> List[ChainResult]:
        """Execute steps in sequence, passing outputs to next step"""
        for step in self.steps:
            result = self._execute_step(step)
            self.results.append(result)
            # Update context with step outputs
            self.context.update(result.output_data)
        return self.results
    
    def _execute_parallel(self) -> List[ChainResult]:
        """Execute independent steps in parallel (simulated)"""
        parallel_results = []
        for step in self.steps:
            if not step.depends_on:  # Independent steps
                result = self._execute_step(step)
                parallel_results.append(result)
        
        # Execute dependent steps after their dependencies
        remaining_steps = [s for s in self.steps if s.depends_on]
        while remaining_steps:
            for step in remaining_steps[:]:
                if all(dep in [r.step_name for r in parallel_results] for dep in step.depends_on):
                    result = self._execute_step(step)
                    parallel_results.append(result)
                    remaining_steps.remove(step)
        
        self.results = parallel_results
        return self.results
    
    def _execute_conditional(self) -> List[ChainResult]:
        """Execute steps based on conditions"""
        for step in self.steps:
            if step.condition:
                # Simple condition evaluation (in real implementation, use proper evaluation)
                if self._evaluate_condition(step.condition):
                    result = self._execute_step(step)
                    self.results.append(result)
                    self.context.update(result.output_data)
            else:
                result = self._execute_step(step)
                self.results.append(result)
                self.context.update(result.output_data)
        return self.results
    
    def _execute_feedback_loop(self, max_iterations: int = 3) -> List[ChainResult]:
        """Execute with feedback loops for refinement"""
        iteration = 0
        while iteration < max_iterations:
            for step in self.steps:
                result = self._execute_step(step)
                self.results.append(result)
                self.context.update(result.output_data)
                
                # Check if refinement is needed
                if "needs_refinement" in result.output_data and result.output_data["needs_refinement"]:
                    self.context["feedback"] = f"Iteration {iteration + 1}: Refining based on previous output"
                    continue
                else:
                    return self.results
            iteration += 1
        return self.results
    
    def _execute_step(self, step: ChainStep) -> ChainResult:
        """Execute a single step in the chain"""
        # Prepare input data for this step
        input_data = {}
        for input_key in step.inputs:
            if input_key in self.context:
                input_data[input_key] = self.context[input_key]
        
        # Format prompt with available context
        prompt = step.prompt_template.format(**self.context)
        
        # Get LLM response
        response = self.llm.invoke(prompt)
        
        # Parse outputs (simplified - in real implementation, use structured parsing)
        output_data = self._parse_outputs(response, step.outputs)
        
        return ChainResult(
            step_name=step.name,
            input_data=input_data,
            output_data=output_data,
            prompt_used=prompt,
            response=response
        )
    
    def _parse_outputs(self, response: str, expected_outputs: List[str]) -> Dict[str, Any]:
        """Parse LLM response to extract expected outputs"""
        # Simplified parsing - in real implementation, use structured output
        outputs = {}
        for output_key in expected_outputs:
            if output_key == "summary":
                outputs[output_key] = response[:100] + "..." if len(response) > 100 else response
            elif output_key == "recommendations":
                outputs[output_key] = [line.strip() for line in response.split('\n') if line.strip().startswith('-')]
            elif output_key == "confidence":
                # Mock confidence score
                outputs[output_key] = 0.85
            elif output_key == "needs_refinement":
                outputs[output_key] = "refine" in response.lower() or "improve" in response.lower()
            else:
                outputs[output_key] = response
        return outputs
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition against current context"""
        # Simplified condition evaluation
        if "confidence > 0.7" in condition:
            return self.context.get("confidence", 0.5) > 0.7
        elif "high_risk" in condition:
            return "risk" in str(self.context).lower()
        return True

# Pre-built chain templates
class ChainTemplates:
    @staticmethod
    def content_creation_chain():
        """Multi-step content creation workflow"""
        chain = PromptChain(get_llm(), ChainType.SEQUENTIAL)
        
        # Step 1: Research and outline
        chain.add_step(ChainStep(
            name="research_outline",
            prompt_template="""Research and create an outline for: {topic}
            Target audience: {audience}
            Content type: {content_type}
            
            Provide:
            1. Key points to cover
            2. Logical structure
            3. Research gaps to address""",
            inputs=["topic", "audience", "content_type"],
            outputs=["outline", "key_points", "research_gaps"]
        ))
        
        # Step 2: First draft
        chain.add_step(ChainStep(
            name="first_draft",
            prompt_template="""Create a first draft based on this outline:
            {outline}
            
            Key points to include: {key_points}
            Audience: {audience}
            
            Focus on clarity and structure.""",
            inputs=["outline", "key_points", "audience"],
            outputs=["draft", "word_count"]
        ))
        
        # Step 3: Review and refine
        chain.add_step(ChainStep(
            name="review_refine",
            prompt_template="""Review this draft and suggest improvements:
            {draft}
            
            Check for:
            - Clarity and flow
            - Audience appropriateness
            - Missing information
            - Tone consistency
            
            Provide specific suggestions.""",
            inputs=["draft", "audience"],
            outputs=["feedback", "suggestions", "confidence"]
        ))
        
        return chain
    
    @staticmethod
    def decision_analysis_chain():
        """Multi-perspective decision analysis"""
        chain = PromptChain(get_llm(), ChainType.PARALLEL)
        
        # Parallel analysis from different perspectives
        perspectives = [
            ("financial", "Analyze from financial/ROI perspective"),
            ("risk", "Identify and assess risks"),
            ("operational", "Consider operational implications"),
            ("strategic", "Evaluate strategic alignment")
        ]
        
        for perspective, instruction in perspectives:
            chain.add_step(ChainStep(
                name=f"{perspective}_analysis",
                prompt_template=f"""Decision to analyze: {{decision}}
                Context: {{context}}
                
                {instruction}:
                {{decision}}
                
                Provide analysis, pros/cons, and recommendations.""",
                inputs=["decision", "context"],
                outputs=[f"{perspective}_analysis", f"{perspective}_score"]
            ))
        
        # Synthesis step (depends on all analyses)
        chain.add_step(ChainStep(
            name="synthesis",
            prompt_template="""Synthesize these analyses into a final recommendation:
            
            Financial: {financial_analysis}
            Risk: {risk_analysis} 
            Operational: {operational_analysis}
            Strategic: {strategic_analysis}
            
            Provide:
            1. Overall recommendation
            2. Key trade-offs
            3. Next steps""",
            inputs=["financial_analysis", "risk_analysis", "operational_analysis", "strategic_analysis"],
            outputs=["recommendation", "trade_offs", "next_steps"],
            depends_on=["financial_analysis", "risk_analysis", "operational_analysis", "strategic_analysis"]
        ))
        
        return chain
    
    @staticmethod
    def adaptive_learning_chain():
        """Adaptive chain that improves based on feedback"""
        chain = PromptChain(get_llm(), ChainType.FEEDBACK_LOOP)
        
        chain.add_step(ChainStep(
            name="initial_attempt",
            prompt_template="""Task: {task}
            Requirements: {requirements}
            Previous feedback: {feedback}
            
            Provide a solution that addresses the requirements.""",
            inputs=["task", "requirements", "feedback"],
            outputs=["solution", "confidence", "needs_refinement"]
        ))
        
        chain.add_step(ChainStep(
            name="self_evaluate",
            prompt_template="""Evaluate this solution: {solution}
            Against requirements: {requirements}
            
            Rate quality (1-10) and identify areas for improvement.
            If score < 8, suggest refinements.""",
            inputs=["solution", "requirements"],
            outputs=["evaluation_score", "improvement_areas", "needs_refinement"]
        ))
        
        return chain

def demo_sequential_chain():
    """Demonstrate sequential chaining"""
    print("=== Sequential Chain Demo: Content Creation ===")
    
    chain = ChainTemplates.content_creation_chain()
    
    initial_data = {
        "topic": "Prompt Engineering Best Practices",
        "audience": "Software engineers new to AI",
        "content_type": "Technical blog post"
    }
    
    results = chain.execute(initial_data)
    
    for result in results:
        print(f"\nStep: {result.step_name}")
        print(f"Prompt: {result.prompt_used[:100]}...")
        print(f"Response: {result.response[:150]}...")
        print(f"Outputs: {list(result.output_data.keys())}")

def demo_parallel_chain():
    """Demonstrate parallel chaining"""
    print("\n=== Parallel Chain Demo: Decision Analysis ===")
    
    chain = ChainTemplates.decision_analysis_chain()
    
    initial_data = {
        "decision": "Should we adopt AI code generation tools company-wide?",
        "context": "Mid-size software company, 200 developers, current productivity concerns"
    }
    
    results = chain.execute(initial_data)
    
    print(f"Executed {len(results)} analysis steps in parallel")
    for result in results:
        if "synthesis" in result.step_name:
            print(f"\nFinal Recommendation: {result.response[:200]}...")

def demo_conditional_chain():
    """Demonstrate conditional chaining"""
    print("\n=== Conditional Chain Demo ===")
    
    chain = PromptChain(get_llm(), ChainType.CONDITIONAL)
    
    # Basic step
    chain.add_step(ChainStep(
        name="risk_assessment",
        prompt_template="Assess risks for: {proposal}",
        inputs=["proposal"],
        outputs=["risk_level", "confidence"]
    ))
    
    # Conditional step - only if high risk
    chain.add_step(ChainStep(
        name="detailed_risk_analysis",
        prompt_template="Perform detailed risk analysis: {proposal}. Previous assessment: {risk_level}",
        inputs=["proposal", "risk_level"],
        outputs=["detailed_risks", "mitigation_strategies"],
        condition="high_risk in context"
    ))
    
    # Always execute final step
    chain.add_step(ChainStep(
        name="final_recommendation",
        prompt_template="Final recommendation for: {proposal}. Risk info: {risk_level}",
        inputs=["proposal", "risk_level"],
        outputs=["recommendation"]
    ))
    
    initial_data = {"proposal": "Implement AI system with access to customer data"}
    results = chain.execute(initial_data)
    
    print(f"Executed {len(results)} conditional steps")
    for result in results:
        print(f"- {result.step_name}: {result.response[:100]}...")

def demo_feedback_loop():
    """Demonstrate feedback loop chaining"""
    print("\n=== Feedback Loop Demo ===")
    
    chain = ChainTemplates.adaptive_learning_chain()
    
    initial_data = {
        "task": "Write a function to validate email addresses",
        "requirements": "Must handle international domains, return detailed error messages, be performant",
        "feedback": ""
    }
    
    results = chain.execute(initial_data)
    
    print(f"Completed {len(results)} iterations with feedback")
    final_result = results[-1]
    print(f"Final solution confidence: {final_result.output_data.get('confidence', 'N/A')}")

def run_all_demos():
    """Run all chaining demonstrations"""
    print("🔗 PROMPT CHAINING WORKFLOWS DEMO")
    print("=" * 50)
    
    demo_sequential_chain()
    demo_parallel_chain() 
    demo_conditional_chain()
    demo_feedback_loop()
    
    print("\n" + "=" * 50)
    print("✅ All chaining patterns demonstrated!")
    print("\nKey Benefits:")
    print("- Break complex tasks into manageable steps")
    print("- Reuse and compose prompt patterns")
    print("- Handle dependencies and conditions")
    print("- Enable feedback and refinement loops")
    print("- Scale to enterprise workflow automation")

if __name__ == "__main__":
    run_all_demos()
