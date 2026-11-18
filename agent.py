import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI

from src.rag.rag_engine import RAGEngine
from src.rag.feedback_engine import TrainingFeedbackEngine

load_dotenv()


class PersonalTrainerAgent:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3):
        """
        Initializes the AI Personal Trainer agent with RAG, Feedback Engine, and Memory.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.rag_engine = RAGEngine()
        self.feedback_engine = TrainingFeedbackEngine()
        
        # In-memory store for session histories (can be swapped for Redis/SQLite in production)
        self.store: Dict[str, BaseChatMessageHistory] = {}
        
        self.prompt = self._build_prompt_template()
        self.chain_with_history = self._build_agent_chain()

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieves or initializes chat history for a given user session."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Defines the specialized Sports Science & High-Performance Coach persona."""
        system_message = """
        You are an elite AI Personal Trainer and Sports Scientist grounded in ACSM, NSCA, 
        and evidence-based hypertrophy/biomechanics literature.

        CORE DIRECTIVES:
        1. Base all exercise selection, volume, intensity, and nutrition recommendations on the retrieved Scientific Context.
        2. Respect biomechanical constraints (e.g., substituting lying leg curls for seated/standing variations when stature or alignment requires it).
        3. When Autoregulation/Feedback data is present, apply the exact load/volume adjustments determined by the system and explain the physiological rationale (RPE, RIR, central/peripheral fatigue) clearly to the athlete.
        4. Be direct, motivating, technical yet accessible, and prioritize long-term injury prevention and progressive overload.

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
        """Connects the prompt template to the LLM and wraps it with session memory."""
        base_chain = self.prompt | self.llm
        return RunnableWithMessageHistory(
            base_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def chat(
        self, 
        user_input: str, 
        session_id: str = "default_athlete",
        session_feedback: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Main entry point for interacting with the Personal Trainer Agent.
        
        Args:
            user_input: The athlete's question or check-in message.
            session_id: Unique identifier to maintain conversation state.
            session_feedback: Optional structured post-workout metrics for autoregulation.
        """
        # 1. Retrieve relevant sports science literature from ChromaDB
        retrieved_docs = self.rag_engine.retrieve(user_input, top_k=4)
        context_text = "\n\n".join(retrieved_docs) if retrieved_docs else "No specific context retrieved."

        # 2. Process post-workout feedback if provided
        if session_feedback:
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

        # 3. Invoke the chain with memory
        response = self.chain_with_history.invoke(
            {
                "input": user_input,
                "context": context_text,
                "feedback_context": feedback_text
            },
            config={"configurable": {"session_id": session_id}}
        )

        return response.content


if __name__ == "__main__":
    # Quick architectural smoke test
    agent = PersonalTrainerAgent()
    
    sample_telemetry = {
        "exercise": "Barbell Squat",
        "target_rir": 2,
        "actual_rir": 0,
        "rpe": 10.0,
        "fatigue_level": "extreme",
        "completed_all_sets": True,
        "joint_discomfort": False
    }
    
    reply = agent.chat(
        user_input="Finished my leg session today, feeling completely drained after squats. How should we adjust next week?",
        session_id="athlete_01",
        session_feedback=sample_telemetry
    )
    print(reply)