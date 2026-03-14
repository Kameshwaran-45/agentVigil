"""
Adaptive Decision Router — Stage 2 (LangGraph Implementation)
===============================================================
WHAT:  Implements the 3-tier Adaptive Decision Router as a LangGraph
       state graph with LLM-powered reasoning at each node.

WHY LANGGRAPH:
  - Graph-based: maps 1:1 to our architecture diagram
  - Deterministic: same input → same routing decision
  - State machine: carries context across all nodes
  - Tool use: agent calls DB tools based on its own reasoning
  - Auditable: every node transition is logged
  - Human-in-loop: can add approval nodes for high-severity alerts

THE GRAPH:
  evaluate_complexity → route_to_tier
    ├── tier1_keyword (LOW)     → synthesize
    ├── tier2_semantic (MEDIUM) → synthesize
    └── tier3_temporal (HIGH)   → synthesize
  synthesize → alert_decision → END

TOOLS THE AGENT CAN USE:
  1. search_similar()      — Milvus vector search
  2. keyword_search()      — PostgreSQL keyword query
  3. get_video_chunks()    — PostgreSQL temporal context
  4. get_event_stats()     — PostgreSQL pattern detection

CONNECTS TO: database.py provides the tools
             perception.py provides captions
             app.py receives the graph output
"""

import time
import json
import operator
from typing import Dict, List, Any, Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import (
    TIER1_THRESHOLD, TIER2_THRESHOLD,
    ALERT_THRESHOLD, WATCH_THRESHOLD,
    ALL_CRIME_KEYWORDS, EVENT_KEYWORDS,
    EVENT_CATEGORIES, SIMILAR_INCIDENT_TOP_K,
)


# ═════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    State that flows through the entire LangGraph.
    Every node reads from and writes to this state.
    """
    # Input (set once at entry)
    video_name: str
    chunk_index: int
    caption: str
    embedding: List[float]
    pg_id: int

    # Complexity evaluation
    complexity_score: float
    tier_route: str                    # "tier1", "tier2", "tier3"
    keyword_hits: int
    matched_events: List[str]

    # Tier results (accumulated as we escalate)
    tier1_score: float
    tier1_event: str
    tier1_evidence: str

    tier2_score: float
    tier2_similar: List[Dict]
    tier2_evidence: str

    tier3_score: float
    tier3_narrative: str
    tier3_evidence: str

    # Final output
    final_score: float
    final_severity: str
    final_alert_status: str
    final_event_type: str
    final_narrative: str
    final_evidence: str
    final_recommendation: str
    similar_count: int
    tier_used: int
    processing_time: float


# ═════════════════════════════════════════════════════════════════════
# GRAPH NODES (each is a function that takes state, returns state)
# ═════════════════════════════════════════════════════════════════════

class AdaptiveReasoningAgent:
    """
    LangGraph-based Adaptive Decision Router.
    Maps directly to Stage 2 in the architecture diagram.
    """

    def __init__(self, database, embedding_engine):
        self.db = database
        self.embedder = embedding_engine
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph.
        This IS the Adaptive Decision Router diamond in the architecture.
        """
        graph = StateGraph(AgentState)

        # ── Add nodes ───────────────────────────────────────────────
        graph.add_node("evaluate_complexity", self._node_evaluate_complexity)
        graph.add_node("tier1_keyword", self._node_tier1)
        graph.add_node("tier2_semantic", self._node_tier2)
        graph.add_node("tier3_temporal", self._node_tier3)
        graph.add_node("synthesize", self._node_synthesize)
        graph.add_node("alert_decision", self._node_alert_decision)

        # ── Set entry point ─────────────────────────────────────────
        graph.set_entry_point("evaluate_complexity")

        # ── Conditional routing (the "diamond" in the diagram) ──────
        graph.add_conditional_edges(
            "evaluate_complexity",
            self._route_to_tier,
            {
                "tier1": "tier1_keyword",
                "tier2": "tier2_semantic",
                "tier3": "tier3_temporal",
            },
        )

        # ── Tier 1 can escalate or go to synthesis ──────────────────
        graph.add_conditional_edges(
            "tier1_keyword",
            self._should_escalate_from_tier1,
            {
                "escalate": "tier2_semantic",
                "synthesize": "synthesize",
            },
        )

        # ── Tier 2 can escalate or go to synthesis ──────────────────
        graph.add_conditional_edges(
            "tier2_semantic",
            self._should_escalate_from_tier2,
            {
                "escalate": "tier3_temporal",
                "synthesize": "synthesize",
            },
        )

        # ── Tier 3 always goes to synthesis ─────────────────────────
        graph.add_edge("tier3_temporal", "synthesize")

        # ── Synthesis → Alert Decision → END ────────────────────────
        graph.add_edge("synthesize", "alert_decision")
        graph.add_edge("alert_decision", END)

        return graph.compile()

    # ═══════════════════════════════════════════════════════════════
    # NODE: Evaluate Complexity (Rule Complexity Evaluator)
    # ═══════════════════════════════════════════════════════════════

    def _node_evaluate_complexity(self, state: AgentState) -> Dict:
        """
        Rule Complexity Evaluator from the architecture diagram.
        Examines the caption and determines initial routing tier.
        """
        caption_lower = state["caption"].lower()

        keyword_hits = sum(1 for kw in ALL_CRIME_KEYWORDS if kw in caption_lower)

        matched_events = []
        for event_type, keywords in EVENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in caption_lower)
            if hits > 0:
                matched_events.append(event_type)

        # Determine complexity
        if keyword_hits == 0:
            complexity = 0.1
            route = "tier1"
        elif keyword_hits <= 2 and not any(
            e in matched_events for e in [
                "Robbery / Armed Robbery",
                "Arson / Fire",
                "Fighting / Assault",
            ]
        ):
            complexity = 0.4
            route = "tier2"
        else:
            complexity = 0.7
            route = "tier3"

        return {
            "complexity_score": round(complexity, 3),
            "tier_route": route,
            "keyword_hits": keyword_hits,
            "matched_events": matched_events,
        }

    # ═══════════════════════════════════════════════════════════════
    # ROUTING FUNCTIONS (conditional edges)
    # ═══════════════════════════════════════════════════════════════

    def _route_to_tier(self, state: AgentState) -> str:
        """The Adaptive Decision Router — decides which tier to start."""
        return state["tier_route"]

    def _should_escalate_from_tier1(self, state: AgentState) -> str:
        """After Tier 1, decide if we need deeper analysis."""
        if state["tier1_score"] >= TIER1_THRESHOLD:
            return "escalate"
        return "synthesize"

    def _should_escalate_from_tier2(self, state: AgentState) -> str:
        """After Tier 2, decide if we need temporal reasoning."""
        if state["tier2_score"] >= TIER2_THRESHOLD:
            return "escalate"
        if state.get("tier2_similar") and len(state["tier2_similar"]) >= 2:
            return "escalate"
        return "synthesize"

    # ═══════════════════════════════════════════════════════════════
    # NODE: Tier 1 — Keyword Retrieval (Low Complexity)
    # ═══════════════════════════════════════════════════════════════

    def _node_tier1(self, state: AgentState) -> Dict:
        """Fast keyword-based analysis. No DB calls."""
        caption_lower = state["caption"].lower()

        best_event = "Normal Activity"
        best_hits = 0
        for event_type, keywords in EVENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in caption_lower)
            if hits > best_hits:
                best_hits = hits
                best_event = event_type

        score = min(state["keyword_hits"] * 0.15, 0.5)

        return {
            "tier1_score": round(score, 3),
            "tier1_event": best_event if state["keyword_hits"] > 0 else "Normal Activity",
            "tier1_evidence": (
                f"Tier 1 Keyword Analysis: {state['keyword_hits']} crime keywords detected. "
                f"Best match: {best_event} ({best_hits} hits)."
            ),
            "tier_used": 1,
        }

    # ═══════════════════════════════════════════════════════════════
    # NODE: Tier 2 — Semantic Search (Moderate Complexity)
    # Uses Milvus TOOL to find similar past incidents
    # ═══════════════════════════════════════════════════════════════

    def _node_tier2(self, state: AgentState) -> Dict:
        """
        Milvus semantic search — TOOL CALL.
        Agent queries vector DB for similar past incidents.
        """
        # First run Tier 1 if we jumped directly here
        if not state.get("tier1_score"):
            tier1 = self._node_tier1(state)
            state.update(tier1)

        # TOOL: Search Milvus for similar incidents
        similar = self.db.search_similar(
            state["embedding"],
            top_k=SIMILAR_INCIDENT_TOP_K,
            exclude_video=state["video_name"],
        )

        base_score = state.get("tier1_score", 0.3)

        if similar:
            past_anomalies = [s for s in similar if s.get("anomaly_score", 0) > 0.5]
            if past_anomalies:
                base_score = min(base_score + 0.25, 0.9)
            elif similar[0]["similarity"] > 0.7:
                base_score = min(base_score + 0.1, 0.7)

        evidence_parts = [state.get("tier1_evidence", "")]
        if similar:
            evidence_parts.append(
                f"Tier 2 Semantic Search: Found {len(similar)} similar past incidents."
            )
            for s in similar[:3]:
                evidence_parts.append(
                    f"  → {s['video_name']} [{s['event_type']}] "
                    f"(similarity: {s['similarity']:.3f}): "
                    f"\"{s.get('caption', '')[:80]}...\""
                )
        else:
            evidence_parts.append("Tier 2 Semantic Search: No similar past incidents found.")

        return {
            "tier2_score": round(base_score, 3),
            "tier2_similar": similar,
            "tier2_evidence": "\n".join(evidence_parts),
            "tier_used": 2,
        }

    # ═══════════════════════════════════════════════════════════════
    # NODE: Tier 3 — Temporal Reasoning + LLM (High Complexity)
    # Uses PostgreSQL TOOL to get neighboring chunks
    # ═══════════════════════════════════════════════════════════════

    def _node_tier3(self, state: AgentState) -> Dict:
        """
        Full temporal reasoning across neighboring chunks.
        Uses PostgreSQL TOOL for temporal context.
        This is the most expensive tier — only for confirmed anomalies.
        """
        # Run Tier 1 + 2 if we jumped directly here
        if not state.get("tier1_score"):
            state.update(self._node_tier1(state))
        if not state.get("tier2_score"):
            state.update(self._node_tier2(state))

        # TOOL: Get neighboring chunks from PostgreSQL
        all_chunks = self.db.get_video_chunks(state["video_name"])
        chunk_index = state["chunk_index"]
        window = 3

        temporal = []
        for c in all_chunks:
            if abs(c["chunk_index"] - chunk_index) <= window:
                temporal.append({
                    "chunk": c["chunk_index"],
                    "time": f"{c['start_sec']}s-{c['end_sec']}s",
                    "caption": c["vlm_caption"],
                    "event": c["event_type"],
                })

        event_chunks = [t for t in temporal if t["event"] != "Normal Activity"]
        normal_before = [
            t for t in temporal
            if t["chunk"] < chunk_index and t["event"] == "Normal Activity"
        ]

        base_score = state.get("tier2_score", 0.5)
        if normal_before and event_chunks:
            base_score = min(base_score + 0.1, 0.95)
        if len(event_chunks) >= 3:
            base_score = min(base_score + 0.15, 0.98)

        # Build temporal narrative
        narrative_parts = []
        if normal_before:
            narrative_parts.append(
                f"Pre-incident: Normal activity in chunks "
                f"{[t['chunk'] for t in normal_before]}."
            )
        for ec in event_chunks:
            narrative_parts.append(
                f"Chunk {ec['chunk']} ({ec['time']}): {ec['caption'][:120]}"
            )

        similar = state.get("tier2_similar", [])
        if similar:
            narrative_parts.append(
                f"\nHistorical: {len(similar)} similar past incidents detected."
            )

        evidence = state.get("tier2_evidence", "")
        evidence += (
            f"\nTier 3 Temporal Analysis: {len(temporal)} chunks in window, "
            f"{len(event_chunks)} anomalous."
        )

        return {
            "tier3_score": round(base_score, 3),
            "tier3_narrative": "\n".join(narrative_parts),
            "tier3_evidence": evidence,
            "tier_used": 3,
        }

    # ═══════════════════════════════════════════════════════════════
    # NODE: Synthesize (combines all tier results)
    # ═══════════════════════════════════════════════════════════════

    def _node_synthesize(self, state: AgentState) -> Dict:
        """Combine results from whichever tiers ran."""
        tier_used = state.get("tier_used", 1)

        if tier_used >= 3 and state.get("tier3_score"):
            score = state["tier3_score"]
            narrative = state.get("tier3_narrative", "")
            evidence = state.get("tier3_evidence", "")
        elif tier_used >= 2 and state.get("tier2_score"):
            score = state["tier2_score"]
            narrative = ""
            evidence = state.get("tier2_evidence", "")
        else:
            score = state.get("tier1_score", 0.0)
            narrative = ""
            evidence = state.get("tier1_evidence", "")

        event_type = state.get("tier1_event", "Normal Activity")
        similar = state.get("tier2_similar", [])

        return {
            "final_score": score,
            "final_event_type": event_type,
            "final_narrative": narrative,
            "final_evidence": evidence,
            "similar_count": len(similar),
        }

    # ═══════════════════════════════════════════════════════════════
    # NODE: Alert Decision (final routing)
    # ═══════════════════════════════════════════════════════════════

    def _node_alert_decision(self, state: AgentState) -> Dict:
        """Final NORMAL/WATCH/ALERT decision + recommendation."""
        score = state["final_score"]

        if score >= ALERT_THRESHOLD:
            severity = "HIGH"
            alert_status = "ALERT"
        elif score >= WATCH_THRESHOLD:
            severity = "MEDIUM"
            alert_status = "WATCH"
        else:
            severity = "LOW"
            alert_status = "NORMAL"

        recommendation = self._generate_recommendation(
            state["final_event_type"], severity, state.get("similar_count", 0)
        )

        return {
            "final_severity": severity,
            "final_alert_status": alert_status,
            "final_recommendation": recommendation,
        }

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API (called by app.py)
    # ═══════════════════════════════════════════════════════════════

    def analyze_chunk(
        self,
        video_name: str,
        chunk_index: int,
        caption: str,
        embedding: List[float],
        pg_id: int,
    ) -> Dict[str, Any]:
        """
        Main entry point — runs the LangGraph state machine.
        Returns structured analysis result.
        """
        start = time.time()

        # Build initial state
        initial_state = {
            "video_name": video_name,
            "chunk_index": chunk_index,
            "caption": caption,
            "embedding": embedding,
            "pg_id": pg_id,
            "complexity_score": 0.0,
            "tier_route": "tier1",
            "keyword_hits": 0,
            "matched_events": [],
            "tier1_score": 0.0,
            "tier1_event": "Normal Activity",
            "tier1_evidence": "",
            "tier2_score": 0.0,
            "tier2_similar": [],
            "tier2_evidence": "",
            "tier3_score": 0.0,
            "tier3_narrative": "",
            "tier3_evidence": "",
            "final_score": 0.0,
            "final_severity": "LOW",
            "final_alert_status": "NORMAL",
            "final_event_type": "Normal Activity",
            "final_narrative": "",
            "final_evidence": "",
            "final_recommendation": "",
            "similar_count": 0,
            "tier_used": 1,
            "processing_time": 0.0,
        }

        # RUN THE GRAPH
        result = self.graph.invoke(initial_state)

        processing_time = time.time() - start

        # Build output dict (same interface as before — app.py unchanged)
        output = {
            "chunk_index": chunk_index,
            "anomaly_score": result["final_score"],
            "severity": result["final_severity"],
            "alert_status": result["final_alert_status"],
            "event_type": result["final_event_type"],
            "tier_used": result["tier_used"],
            "narrative": result["final_narrative"],
            "evidence": result["final_evidence"],
            "recommendation": result["final_recommendation"],
            "similar_count": result["similar_count"],
            "similar_incidents": result.get("tier2_similar", []),
            "processing_time": round(processing_time, 3),
        }

        # Update database
        self.db.update_chunk_analysis(
            pg_id=pg_id,
            anomaly_score=output["anomaly_score"],
            severity=output["severity"],
            alert_status=output["alert_status"],
            tier_used=output["tier_used"],
            narrative=output["narrative"],
            evidence=output["evidence"],
            recommendation=output["recommendation"],
            similar_count=output["similar_count"],
        )

        return output

    def analyze_full_video(
        self, video_name: str, duration_sec: float,
    ) -> Dict[str, Any]:
        """Video-level analysis after all chunks processed."""
        chunks = self.db.get_video_chunks(video_name)
        if not chunks:
            return {"error": "No chunks found"}

        total = len(chunks)
        anomalous = [
            c for c in chunks if c["alert_status"] in ("ALERT", "WATCH")
        ]

        event_counts = {}
        for c in anomalous:
            et = c["event_type"]
            event_counts[et] = event_counts.get(et, 0) + 1
        primary_event = (
            max(event_counts, key=event_counts.get)
            if event_counts else "Normal Activity"
        )

        scores = [c["anomaly_score"] for c in chunks]
        overall_score = max(scores) if scores else 0.0

        if overall_score >= ALERT_THRESHOLD:
            overall_severity = "HIGH"
            overall_alert = "ALERT"
        elif overall_score >= WATCH_THRESHOLD:
            overall_severity = "MEDIUM"
            overall_alert = "WATCH"
        else:
            overall_severity = "LOW"
            overall_alert = "NORMAL"

        narrative_parts = [
            f"Video: {video_name} | Duration: {duration_sec}s | "
            f"{total} chunks analyzed",
            f"Result: {len(anomalous)} anomalous, "
            f"{total - len(anomalous)} normal",
            f"Primary event: {primary_event} | Peak score: {overall_score}",
        ]
        if anomalous:
            narrative_parts.append("\nTimeline:")
            for c in anomalous:
                narrative_parts.append(
                    f"  [{c['start_sec']}s-{c['end_sec']}s] "
                    f"{c['event_type']} (score: {c['anomaly_score']}, "
                    f"tier: {c['tier_used']})"
                )

        narrative = "\n".join(narrative_parts)

        self.db.store_video_summary(
            video_name=video_name,
            duration_sec=duration_sec,
            total_chunks=total,
            anomalous_chunks=len(anomalous),
            primary_event=primary_event,
            overall_score=overall_score,
            overall_severity=overall_severity,
            overall_alert=overall_alert,
            narrative=narrative,
        )

        return {
            "video_name": video_name,
            "total_chunks": total,
            "anomalous_chunks": len(anomalous),
            "primary_event": primary_event,
            "overall_score": round(overall_score, 3),
            "overall_severity": overall_severity,
            "overall_alert": overall_alert,
            "narrative": narrative,
            "chunk_details": [
                {
                    "chunk": c["chunk_index"],
                    "time": f"{c['start_sec']}s-{c['end_sec']}s",
                    "event": c["event_type"],
                    "score": c["anomaly_score"],
                    "severity": c["severity"],
                    "tier": c["tier_used"],
                }
                for c in chunks
            ],
        }

    def _generate_recommendation(self, event_type, severity, similar_count):
        recs = {
            "Road Accident / Vehicle Collision":
                "Dispatch emergency services. Secure area. Preserve footage.",
            "Robbery / Armed Robbery":
                "Alert law enforcement immediately. Lock down exits.",
            "Fighting / Assault":
                "Alert security. Dispatch to location. File report.",
            "Shoplifting / Stealing":
                "Alert loss prevention. Track suspect. Preserve footage.",
            "Vandalism / Property Damage":
                "Alert maintenance and security. Document damage.",
            "Arson / Fire":
                "Trigger fire alarm. Evacuate. Dispatch fire services.",
            "Normal Activity":
                "No action required. Continue monitoring.",
        }
        rec = recs.get(event_type, "Review footage manually.")
        if similar_count >= 3:
            rec += (
                f" ⚠️ PATTERN: {similar_count} similar past incidents "
                f"— consider preventive measures."
            )
        return rec