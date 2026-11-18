"""
Core Agent Orchestrator for the AI Personal Trainer.
Coordinates RAG literature retrieval, post-workout feedback autoregulation,
persistent memory, and LLM response synthesis.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from projetoia.core.rag import RAGEngine
from projetoia.database.memory import AthleteMemoryManager
from projetoia.tests.feedback_agent import TrainingFeedbackEngine

# Internal project modules


load_dotenv()


class PersonalTrainerOrchestrator:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.3):
        """
        Initializes the orchestrator with LLM, RAG retriever, feedback engine, and persistent memory.
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.rag_engine = RAGEngine()
        self.feedback_engine = TrainingFeedbackEngine()
        self.memory_manager = AthleteMemoryManager()

        self.prompt = self._build_prompt_template()
        self.chain_with_history = self._build_agent_chain()

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Defines the elite sports science coach persona."""
        system_message = """
        You are an elite AI Personal Trainer and Sports Scientist grounded in ACSM, NSCA, 
        and evidence-based hypertrophy/biomechanics literature.

        CORE DIRECTIVES:
        1. Base all exercise selection, volume, intensity, and nutrition recommendations on the retrieved Scientific Context.
        2. Respect biomechanical constraints (e.g., substituting lying leg curls for seated/standing variations when stature or alignment requires it).
        3. When Autoregulation/Feedback data is present, apply the exact load/volume adjustments determined by the system and explain the physiological rationale clearly to the athlete.
        4. Be direct, motivating, technical yet accessible, and prioritize long-term injury prevention and progressive overload.

        ATHLETE LONG-TERM PROFILE:
        {athlete_profile}

        SCIENTIFIC CONTEXT (RAG):
        {context}

        AUTOREGULATION & FEEDBACK ANALYSIS:
        {feedback_context}
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

    def _build_agent_chain(self) -> RunnableWithMessageHistory:
        """Connects prompt template to LLM and attaches SQLite-backed session memory."""
        base_chain = self.prompt | self.llm
        return RunnableWithMessageHistory(
            base_chain,
            self.memory_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def process_turn(
        self,
        user_input: str,
        session_id: str = "athlete_01",
        session_feedback: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Processes a user turn by gathering RAG context, analyzing feedback telemetry,
        loading the athlete profile, and invoking the LLM chain.
        """
        # 1. Retrieve sports science context from ChromaDB
        retrieved_docs = self.rag_engine.retrieve(user_input, top_k=4)
        context_text = "\n\n".join(retrieved_docs) if retrieved_docs else "No specific literature context retrieved."

        # 2. Load persistent athlete profile
        athlete_profile_text = self.memory_manager.format_profile_for_prompt(session_id)

        # 3. Process post-workout feedback if provided
        if session_feedback:
            # Optional Pydantic validation safeguard
            try:
                validated_feedback = SessionFeedbackInput(**session_feedback)
                analysis = self.feedback_engine.analyze_feedback(validated_feedback.model_dump())
            except Exception:
                analysis = self.feedback_engine.analyze_feedback(session_feedback)

            feedback_text = (
                f"- Exercise: {session_feedback.get('exercise')}\n"
                f"- Reported RPE: {session_feedback.get('rpe')} | Target RIR: {session_feedback.get('target_rir')} vs Actual RIR: {session_feedback.get('actual_rir')}\n"
                f"- Fatigue Level: {session_feedback.get('fatigue_level')}\n"
                f"- System Action: {analysis['action'].upper()}\n"
                f"- Load Modifier: {analysis['load_modifier']:+.0%} | Volume Modifier: {analysis['volume_modifier']:+d} sets\n"
                f"- Technical Rationale: {analysis['reason']}"
            )
        else:
            feedback_text = "No post-workout telemetry submitted in this turn."

        # 4. Invoke chain with memory config
        response = self.chain_with_history.invoke(
            {
                "input": user_input,
                "context": context_text,
                "athlete_profile": athlete_profile_text,
                "feedback_context": feedback_text
            },
            config={"configurable": {"session_id": session_id}}
        )

        return response.content