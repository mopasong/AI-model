import streamlit as st
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.set_page_config(page_title="🧠 Mumyeong DistilGPT2 (Lite)", layout="centered")

@st.cache_resource
def load_distil_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return pipe

pipe = load_distil_pipeline()

LOG_FILE = "distilgpt2_log.json"
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        session_log = json.load(f)
else:
    session_log = []

def get_color_by_emotion(text):
    if any(k in text for k in ["무서워", "싫어", "두려워"]):
        return "#6c5ce7"
    if any(k in text for k in ["그냥", "몰라", "비슷"]):
        return "#b2bec3"
    if any(k in text for k in ["화나", "짜증", "폭발"]):
        return "#d63031"
    if any(k in text for k in ["좋아", "고마워", "다행"]):
        return "#00cec9"
    return "#ffeaa7"

def get_delay():
    if "last_input_time" not in st.session_state:
        st.session_state.last_input_time = datetime.now()
        return 0
    now = datetime.now()
    delay = (now - st.session_state.last_input_time).total_seconds()
    st.session_state.last_input_time = now
    return delay

st.markdown("""
<style>
@keyframes fadeDots {
  0% {opacity: 1;}
  50% {opacity: 0.3;}
  100% {opacity: 0;}
}
.dot-anim {
  font-size: 32px;
  animation: fadeDots 2s infinite;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mumyeong DistilGPT2 (Lite)")

user_input = st.text_input("💬 말해보세요 (또는 감정)", key="user_input")

if user_input:
    delay = get_delay()

    color = get_color_by_emotion(user_input)
    if delay > 5:
        st.markdown("<div style='height:50px; background-color:#dfe6e9;'></div>", unsafe_allow_html=True)
        st.markdown("🤫 **침묵이 감지되었습니다**", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='dot-anim' style='color:{color}'>● ● ●</div>", unsafe_allow_html=True)

    with st.spinner("🤖 DistilGPT2 응답 생성 중..."):
        result = pipe(user_input, max_new_tokens=100, temperature=0.8)[0]["generated_text"]
        st.success(result.strip())

        session_log.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "ai": result.strip()
        })

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(session_log, f, ensure_ascii=False, indent=2)

if session_log:
    st.markdown("### 📊 대화 흐름 시각화")

    timestamps = [e["timestamp"][:19] for e in session_log]
    fig, ax = plt.subplots()
    ax.plot(timestamps, range(len(timestamps)), marker='o', color='#0984e3')
    ax.set_title("시간에 따른 입력 흐름")
    ax.set_xlabel("시간")
    ax.set_ylabel("대화 순서")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.download_button("⬇️ 대화 기록 다운로드", data=json.dumps(session_log, ensure_ascii=False, indent=2), file_name="distilgpt2_log.json", mime="application/json")