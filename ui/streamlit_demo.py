# ui/streamlit_demo.py
# Run: streamlit run ui/streamlit_demo.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

import streamlit as st
import plotly.graph_objects as go
import math

from agents.bayesian_agent import BayesianAgent
from agents.pure_llm_agent import PureLLMAgent
from agents.cpip import compute_entropy, normalized_entropy
from agents.decision_layer import REQUIRED_SLOTS, ACTIONS

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bayesian Cognitive Framework — VIVA Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

if "bayes_agent" not in st.session_state:
    st.session_state.bayes_agent   = BayesianAgent()
    st.session_state.llm_agent     = PureLLMAgent()
    st.session_state.bayes_display = []
    st.session_state.llm_display   = []
    st.session_state.belief_hist   = []
    st.session_state.entropy_hist  = []
    st.session_state.cpip_info     = None
    st.session_state.started       = False
    st.session_state.compare_mode  = True

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Research Demo")
    st.markdown(
        "*Bayesian-Enhanced Cognitive Framework  \n"
        "for Adaptive Conversational AI Agents*"
    )
    st.divider()

    if st.button(
        "🚀 Start New Conversation",
        use_container_width=True,
        type="primary"
    ):
        with st.spinner(
            "Initializing CPIP prior from LLM world knowledge..."
        ):
            init_info = st.session_state.bayes_agent.initialize()
            st.session_state.llm_agent.reset()

        st.session_state.bayes_display = []
        st.session_state.llm_display   = []
        st.session_state.belief_hist   = []
        st.session_state.entropy_hist  = []
        st.session_state.cpip_info     = init_info
        st.session_state.started       = True
        st.rerun()

    st.divider()
    st.session_state.compare_mode = st.toggle(
        "Compare with Pure LLM",
        value=True,
        help="Show both agents side by side"
    )

    st.divider()
    st.markdown("**💬 Example Messages**")
    examples = [
        "Hi, I'd like to book a table",
        "Somewhere in East Village",
        "This Saturday at 7pm",
        "For 4 people please",
        "We prefer outdoor seating",
        "Do you have Carbone?",
        "Those times don't work for us",
        "Yes let's confirm that",
    ]
    for ex in examples:
        if st.button(
            ex, use_container_width=True, key=f"btn_{ex}"
        ):
            st.session_state["prefill"] = ex
            st.rerun()

    st.divider()

    if st.session_state.started:
        st.markdown("**📊 Session Stats**")
        agent = st.session_state.bayes_agent
        if agent.belief_state:
            dom, conf = agent.belief_state.get_dominant()
            st.metric("Turn",
                       agent.belief_state.turn_count)
            st.metric("Dominant Intent",
                       dom.replace("_", " ").title())
            st.metric("Confidence", f"{conf:.1%}")

            confirmed = agent.confirmed_slots
            st.markdown("**Slot Status**")
            for slot in REQUIRED_SLOTS:
                val = confirmed.get(slot)
                if val and val != "pending":
                    st.success(f"✓ {slot}: {val}")
                elif val == "pending":
                    st.warning(f"⏳ {slot}: asking...")
                else:
                    st.error(f"✗ {slot}: needed")

# ══════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════

st.markdown(
    "# 🍽️ Restaurant Booking — Research Demonstration"
)
st.caption(
    "MSc Thesis: Bayesian-Enhanced Cognitive Framework for "
    "Adaptive Conversational AI Agents"
)

# CPIP initialization info
if st.session_state.cpip_info:
    info = st.session_state.cpip_info
    with st.expander(
        f"🧠 CPIP Initialized — "
        f"{info['reduction']}% entropy reduction vs uniform prior",
        expanded=False
    ):
        c1, c2, c3 = st.columns(3)
        c1.metric("Initial Entropy", f"{info['entropy']:.4f}")
        c2.metric("Normalized H",    f"{info['h_norm']:.4f}")
        c3.metric("Uncertainty ↓",   f"{info['reduction']}%")
        st.caption(f"*{info['reasoning']}*")
        st.markdown("**Initial CPIP Prior Distribution:**")
        prior_sorted = sorted(
            info["prior"].items(),
            key=lambda x: x[1], reverse=True
        )
        for cls, prob in prior_sorted:
            bar = "█" * int(prob * 35)
            st.text(
                f"  {cls:<25} {prob:.4f}  {bar}"
            )

if not st.session_state.started:
    st.info(
        "👈 Click **Start New Conversation** to begin. "
        "The system will initialize CPIP prior using "
        "LLM world knowledge."
    )

    st.markdown("---")
    st.markdown("### What this demo proves:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            "**🎯 CPIP**  \n"
            "LLM-bootstrapped priors reduce initial uncertainty "
            "11–18% before conversation starts"
        )
    with col2:
        st.markdown(
            "**📊 Bayesian Updates**  \n"
            "Belief distribution updates mathematically "
            "at every turn using Bayes theorem"
        )
    with col3:
        st.markdown(
            "**⚡ Decision Theory**  \n"
            "Actions selected by expected utility — "
            "not pattern matching"
        )
    with col4:
        st.markdown(
            "**🔍 Explainability**  \n"
            "Every decision fully traceable with "
            "mathematical justification"
        )
    st.stop()

# ══════════════════════════════════════════════════════════════
# CONVERSATION PANELS
# ══════════════════════════════════════════════════════════════

if st.session_state.compare_mode:
    col_b, col_l = st.columns(2)
else:
    col_b = st.container()
    col_l = None

with col_b:
    if st.session_state.compare_mode:
        st.markdown("#### 🧠 Bayesian Cognitive Framework")
        st.caption(
            "✅ CPIP Prior  ✅ Belief Updates  "
            "✅ Decision Theory  ✅ Explainable"
        )
    else:
        st.markdown(
            "#### 🧠 Bayesian Cognitive Framework Agent"
        )

    chat_b = st.container(height=360)
    with chat_b:
        for msg in st.session_state.bayes_display:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    st.caption(
                        f"Turn {msg.get('turn','')} | "
                        f"Action: {msg.get('action','')} | "
                        f"H={msg.get('entropy',0):.3f}"
                    )

if st.session_state.compare_mode and col_l:
    with col_l:
        st.markdown("#### 🤖 Pure LLM Baseline")
        st.caption(
            "❌ No Belief State  ❌ No Explainability  "
            "❌ Black Box"
        )
        chat_l = st.container(height=360)
        with chat_l:
            for msg in st.session_state.llm_display:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

# ── Input ──────────────────────────────────────────────────────
prefill    = st.session_state.pop("prefill", "")
user_input = st.chat_input(
    "Type your message — processed by both agents..."
)
text = user_input or prefill

if text and st.session_state.started:

    with st.spinner("Processing through Bayesian framework..."):
        b_result = st.session_state.bayes_agent.process_turn(
            text
        )

    if st.session_state.compare_mode:
        with st.spinner("Pure LLM responding..."):
            l_result = st.session_state.llm_agent.process_turn(
                text
            )
    else:
        l_result = None

    # Update displays
    st.session_state.bayes_display.extend([
        {"role": "user",      "content": text},
        {
            "role":    "assistant",
            "content": b_result["response"],
            "turn":    b_result["turn"],
            "action":  b_result["action"]["action_id"],
            "entropy": b_result["entropy"]
        }
    ])

    if l_result:
        st.session_state.llm_display.extend([
            {"role": "user",      "content": text},
            {"role": "assistant", "content": l_result["response"]}
        ])

    # Track belief and entropy history
    st.session_state.belief_hist.append({
        "turn":   b_result["turn"],
        "belief": dict(b_result["belief_after"])
    })
    st.session_state.entropy_hist.append({
        "turn":    b_result["turn"],
        "entropy": b_result["entropy"],
        "h_norm":  b_result["h_norm"]
    })

    st.rerun()

# ══════════════════════════════════════════════════════════════
# LIVE BELIEF VISUALIZATION
# ══════════════════════════════════════════════════════════════

if st.session_state.belief_hist:
    st.divider()
    st.markdown("### 📊 Live Cognitive State")

    v1, v2, v3, v4 = st.columns([2.5, 1.5, 1, 1])

    with v1:
        st.markdown(
            "**Belief Distribution p(θ_t | z₁:t)**"
        )
        belief   = st.session_state.bayes_agent.belief_state.belief
        sorted_b = sorted(
            belief.items(), key=lambda x: x[1], reverse=True
        )
        labels = [s[0].replace("_", " ") for s in sorted_b]
        values = [s[1] for s in sorted_b]
        max_v  = max(values)
        colors = [
            "#2ecc71" if v == max_v else
            "#3498db" if v > 0.10 else "#bdc3c7"
            for v in values
        ]
        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside"
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=75, t=5, b=0),
            xaxis=dict(range=[0, 1]),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with v2:
        st.markdown("**Entropy Decay H(θ_t)**")
        if len(st.session_state.entropy_hist) >= 1:
            turns  = [e["turn"]
                      for e in st.session_state.entropy_hist]
            hnorms = [e["h_norm"]
                      for e in st.session_state.entropy_hist]
            fig2 = go.Figure()
            # CPIP initial line
            if st.session_state.cpip_info:
                fig2.add_hline(
                    y=st.session_state.cpip_info["h_norm"],
                    line_dash="dot", line_color="#3498db",
                    annotation_text="CPIP start",
                    annotation_position="right"
                )
            fig2.add_trace(go.Scatter(
                x=turns, y=hnorms,
                mode="lines+markers",
                line=dict(color="#2ecc71", width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.15)"
            ))
            fig2.update_layout(
                height=250,
                margin=dict(l=0, r=60, t=5, b=30),
                xaxis_title="Turn",
                yaxis=dict(range=[0, 1.05],
                           title="Uncertainty")
            )
            st.plotly_chart(fig2, use_container_width=True)

    with v3:
        st.markdown("**Cognitive State**")
        agent = st.session_state.bayes_agent
        if agent.belief_state:
            dom, conf = agent.belief_state.get_dominant()
            h         = st.session_state.entropy_hist[-1]["h_norm"]
            st.metric("Intent",
                       dom.replace("_", " ").title())
            st.metric("Confidence", f"{conf:.1%}")
            st.metric("Uncertainty", f"{h:.1%}")
            st.metric("Turn", agent.belief_state.turn_count)

    with v4:
        st.markdown("**Required Slots**")
        agent     = st.session_state.bayes_agent
        confirmed = agent.confirmed_slots
        for slot in REQUIRED_SLOTS:
            val = confirmed.get(slot)
            if val and val != "pending":
                st.success(f"✓ {slot}")
            elif val == "pending":
                st.warning(f"⏳ {slot}")
            else:
                st.error(f"✗ {slot}")

    # Belief trajectory
    if len(st.session_state.belief_hist) > 1:
        st.markdown(
            "**Sequential Belief Updating — "
            "p(θ_t | z₁:t) Across Conversation**"
        )
        history = st.session_state.belief_hist
        turns   = [h["turn"] for h in history]
        from agents.cpip import BELIEF_CLASSES
        top_cls = sorted(
            BELIEF_CLASSES,
            key=lambda c: max(
                h["belief"].get(c, 0) for h in history
            ),
            reverse=True
        )[:5]

        palette = ["#2ecc71", "#3498db", "#e74c3c",
                   "#f39c12", "#9b59b6"]
        fig3 = go.Figure()
        for i, cls in enumerate(top_cls):
            probs = [h["belief"].get(cls, 0) for h in history]
            fig3.add_trace(go.Scatter(
                x=turns, y=probs,
                mode="lines+markers",
                name=cls.replace("_", " "),
                line=dict(color=palette[i], width=2.5),
                marker=dict(size=7)
            ))
        fig3.update_layout(
            xaxis_title="Conversation Turn",
            yaxis_title="p(θ_t)",
            yaxis=dict(range=[0, 1]),
            height=280,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02
            ),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "p(θ_t | z₁:t) ∝ p(z_t | θ_t) × p(θ_t | z₁:t₋₁)  "
            "— belief evolves mathematically as evidence accumulates"
        )

# ══════════════════════════════════════════════════════════════
# EXPLAINABILITY PANEL
# ══════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 🔍 Explainability — Counter Questions")
st.caption(
    "Ask either agent to explain its reasoning. "
    "This is the key research differentiator."
)

exp1, exp2 = st.columns(2)

with exp1:
    if st.button(
        "🧠 Why did Bayesian agent say that?",
        use_container_width=True, type="primary"
    ):
        agent = st.session_state.bayes_agent
        if agent.reasoning_log:
            last = agent.reasoning_log[-1]
            with st.expander(
                f"Full Reasoning Trace — Turn {last['turn']}",
                expanded=True
            ):
                st.markdown(
                    f"**You said:** \"{last['user_said']}\""
                )

                st.markdown("---")
                st.markdown(
                    "**STEP 1 — Evidence Extraction (z_t):**"
                )
                ev = last["evidence"]
                st.json({
                    "intent":     ev.get("intent"),
                    "confidence": ev.get("confidence"),
                    "sentiment":  ev.get("sentiment"),
                    "slots": {
                        k: v for k, v in
                        ev.get("slots", {}).items() if v
                    },
                    "summary": ev.get("key_information")
                })

                st.markdown(
                    "**STEP 2 — Bayesian Belief Update:**"
                )
                bc1, bc2 = st.columns(2)
                with bc1:
                    st.markdown("*Before:*")
                    top3b = sorted(
                        last["belief_before"].items(),
                        key=lambda x: x[1], reverse=True
                    )[:4]
                    for cls, prob in top3b:
                        st.write(f"• {cls}: {prob:.3f}")
                with bc2:
                    st.markdown("*After:*")
                    top3a = sorted(
                        last["belief_after"].items(),
                        key=lambda x: x[1], reverse=True
                    )[:4]
                    for cls, prob in top3a:
                        st.write(f"• {cls}: {prob:.3f}")

                if last.get("belief_shift"):
                    st.markdown("*Significant shifts:*")
                    for cls, shift in sorted(
                        last["belief_shift"].items(),
                        key=lambda x: abs(x[1]), reverse=True
                    )[:4]:
                        arrow = "↑" if shift > 0 else "↓"
                        st.write(
                            f"{arrow} {cls}: {shift:+.3f}"
                        )

                st.markdown(
                    f"**STEP 3 — Action Selected:**  "
                    f"`{last['action']['action_id']}`"
                )
                st.markdown(
                    f"**Reasoning:** {last['action']['reason']}"
                )

                st.markdown("**Action Scores (all actions):**")
                for scored in last.get("all_scores", [])[:5]:
                    status = "🚫" if scored["blocked"] else "✅"
                    st.write(
                        f"{status} {scored['label']}: "
                        f"{scored['score']:.4f}"
                    )

                st.markdown(
                    f"**STEP 4 — Response Generated:**  \n"
                    f"\"{last['response']}\""
                )
                st.markdown(
                    f"**Entropy:** {last['entropy']:.4f} | "
                    f"H_norm: {last['h_norm']:.3f} | "
                    f"Confidence: {last['confidence']:.1%}"
                )

                if len(agent.reasoning_log) > 1:
                    st.markdown(
                        "**Full belief trajectory:**"
                    )
                    for log in agent.reasoning_log:
                        st.write(
                            f"Turn {log['turn']}: "
                            f"\"{log['user_said'][:30]}\" → "
                            f"{log['dominant']} "
                            f"({log['confidence']:.2f}) | "
                            f"H={log['h_norm']:.2f}"
                        )
        else:
            st.info("Send a message first")

with exp2:
    if st.button(
        "🤖 Why did Pure LLM say that?",
        use_container_width=True
    ):
        with st.expander(
            "Pure LLM — Explainability Response",
            expanded=True
        ):
            st.error(
                "**The Pure LLM cannot explain its reasoning.**"
            )
            st.markdown("""
**Why not?**

| Feature | Bayesian Framework | Pure LLM |
|---|---|---|
| Belief state | ✅ Explicit p(θ) | ❌ None |
| Evidence structure | ✅ Structured z_t | ❌ Raw text |
| Decision trace | ✅ Full audit | ❌ Not available |
| Uncertainty measure | ✅ Shannon entropy | ❌ None |
| Action justification | ✅ Expected utility | ❌ None |

The response was generated by neural network pattern matching.
There is no accessible reasoning process to inspect.

**This is precisely the limitation this research addresses.**
            """)
            st.success(
                "👈 Click the Bayesian button to see the "
                "contrast — full mathematical reasoning trace "
                "available at every turn."
            )