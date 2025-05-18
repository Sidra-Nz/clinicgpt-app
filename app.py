import streamlit as st
import os
import pandas as pd
import gspread

from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI

def setup_llm():
    return ChatOpenAI(
        model_name="deepseek/deepseek-r1-zero:free",  # Or "deepseek-coder:6.7b" etc.
        temperature=0.2,
        openai_api_base="https://openrouter.ai/api/v1",  # Optional if already set in env
        openai_api_key="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"  # Optional if already set in env
    )


# Map weekday names to Python’s weekday numbers
WEEKDAYS = {
    "Monday":    0,
    "Tuesday":   1,
    "Wednesday": 2,
    "Thursday":  3,
    "Friday":    4,
    "Saturday":  5,
    "Sunday":    6,
}


DOCTOR_SCHEDULES = {
    "Dr. Ambreen Atif": {
        "specialty": "Orthodontist",
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "time": "10:00 AM – 2:00 PM"
    },
    "Dr. Faiq Qavi": {
        "specialty": "Endodontist",
        "days": ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        "time": "3:00 PM – 7:00 PM"
    },
    "Dr. M. Shariq": {
        "specialty": "General Dentistry",
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        "time": "8:00 AM – 2:00 PM"
    },
    "Dr. Amir Danish": {
        "specialty": "Periodontist",
        "days": ["Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "time": "11:00 AM – 3:00 PM"
    },
    "Dr. Saba Fatima": {
        "specialty": "Pediatric Dentist",
        "days": ["Tuesday", "Wednesday"],
        "time": "9:00 AM – 1:00 PM"
    },
    "Dr. Ahmed Ali": {
        "specialty": "Oral Pathologist / Oral Surgeon",
        "days": ["Monday"],
        "time": "12:00 AM – 2:00 PM"
    },
    "Dr. Kashif": {
        "specialty": "Prosthodontist",
        "days": ["Monday", "Wednesday", "Friday"],
        "time": "2:00 PM – 5:00 PM"
    },
}


# Load hospital knowledge base
def load_knowledge_base():
    loader = TextLoader("clinic_data.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db.as_retriever()




def get_slots_for_doctor(doctor_name):
    sched     = DOCTOR_SCHEDULES[doctor_name]
    today     = datetime.now()
    slots     = []

    for day_name in sched["days"]:
        target_wd  = WEEKDAYS[day_name]
        current_wd = today.weekday()
        # compute days until next occurrence (at least 1)
        delta_days = (target_wd - current_wd + 7) % 7 or 7
        slot_date  = today + timedelta(days=delta_days)
        # format: "Monday, 19 May 2025 – 10:00 AM – 2:00 PM"
        slots.append(slot_date.strftime(f"%A, %d %B %Y – {sched['time']}"))

    return slots

# Get Google Sheet
def get_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("gcp_credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("ClinicAppointments").sheet1  # Replace with your actual sheet name
    return sheet

# Save to googlesheet
def save_appointment(data):
    sheet = get_gsheet()
    columns = ["Patient_Name", "Patient_ID", "Phone No.", "History", "Doctor", "Appointment Date-Time", "Comment", "Status"]
    row = [data.get(col, "") for col in columns]
    sheet.append_row(row)


# Main App
def main():
    st.set_page_config(page_title="ClinicGPT – City Dental Hospital", layout="centered")
    st.title("ClinicGPT 🦷 – City Dental Hospital")

    # ── Add these 5 lines to show welcome exactly once ──
    ss = st.session_state
    ss.setdefault("messages", [])
    if not ss.get("welcome_shown", False):
        ss.welcome_shown = True
        ss.messages.append({
            "role": "ClinicGPT",
            "content": "👋 Hello! I’m ClinicGPT. I can answer questions about our services or help you book an appointment. Just type above to get started!"
        })
    # ─────────────────────────────────────────────────────





    retriever = load_knowledge_base()
    llm = setup_llm()
    prompt_template = PromptTemplate.from_template(
        """You are ClinicGPT, an intelligent and friendly assistant for City Dental Hospital. Never say that based on given text or information. 
            Always base your answers **only** on the provided context below.
            If the answer cannot be found in the context, reply politely: 
            "I'm sorry, I don't have that information right now."

        Be helpful, empathetic, and professional in tone.

        Question: {question}
        =========
        Context:
        {context}
        =========
        Helpful Answer:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Session state management
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "booking_stage" not in st.session_state:
        st.session_state.booking_stage = "none"
    if "appointment_data" not in st.session_state:
        st.session_state.appointment_data = {}

    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message:", key="chat_input")
        submitted = st.form_submit_button("Send")

        if "asked_for_appointment" not in st.session_state:
            st.session_state.asked_for_appointment = False

        if submitted and user_input.strip():
            st.session_state.messages.append({"role": "User", "content": user_input})

            if st.session_state.booking_stage != "none":
                handle_booking_flow(user_input)

            elif any(kw in user_input.lower() for kw in ["appointment", "book", "another appointment", "schedule", "booking"]):
                st.session_state.booking_stage = "ask_willing"
                st.session_state.asked_for_appointment = True
                st.session_state.appointment_data = {}  # ensure data is reset
                st.session_state.messages.append({
                    "role": "ClinicGPT",
                    "content": "Would you like to book an appointment? (yes/no)"
                })

            elif st.session_state.asked_for_appointment and user_input.lower() in ["yes", "y"]:
                st.session_state.booking_stage = "get_name"
                st.session_state.asked_for_appointment = False
                st.session_state.messages.append({"role": "ClinicGPT", "content": "Great! What is your full name?"})

            elif st.session_state.asked_for_appointment and user_input.lower() in ["no", "n"]:
                st.session_state.booking_stage = "none"
                st.session_state.asked_for_appointment = False
                st.session_state.messages.append({
                    "role": "ClinicGPT",
                    "content": "No problem! Let me know if you need help with anything else."
                })




            else:
                # Normal Q&A flow
                response = qa_chain.run(user_input)
                # Remove LaTeX-style \boxed{} and other math wrappers
                if isinstance(response, str):
                    response = response.replace("\\boxed{", "").replace("}", "")  # handles \boxed{}
                    response = response.replace("$$", "").replace("$", "")       # remove LaTeX math
                    response = response.replace("\\(", "").replace("\\)", "")    # remove inline math
                st.session_state.messages.append({"role": "ClinicGPT", "content": response})

                # Ask for appointment after answering
                if not st.session_state.asked_for_appointment:
                    st.session_state.asked_for_appointment = True
                    st.session_state.messages.append({
                        "role": "ClinicGPT",
                        "content": "Would you like to book an appointment? (yes/no)"
                    })


    # Chat history
    with st.container(height=300):
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "User":
                st.markdown(
                    f"""
                    <div style="background-color:#e6f0ff; padding:10px; border-radius:10px; margin-bottom:5px;">
                        <strong>🧑‍💬 You:</strong><br>{msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color:#f2f2f2; padding:10px; border-radius:10px; margin-bottom:5px;">
                        <strong>🤖 ClinicGPT:</strong><br>{msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )



def handle_booking_flow(user_input):
    stage = st.session_state.booking_stage
    data = st.session_state.appointment_data

    def bot(msg):
        msg_with_tip = f"{msg}\n\n_Type 'skip' to exit the appointment flow._"
        st.session_state.messages.append({"role": "ClinicGPT", "content": msg_with_tip})

    # ✅ Allow skipping
    if user_input.lower() in ["skip", "exit", "cancel"]:
        st.session_state.booking_stage = "none"
        st.session_state.appointment_data = {}
        bot("Appointment flow has been cancelled. Let me know if you need anything else.")
        return



    if stage == "ask_willing":
        if user_input.lower() in ["yes", "y"]:
            st.session_state.booking_stage = "get_name"
            bot("Great! What is your full name?")
        else:
            st.session_state.booking_stage = "none"
            bot("No problem! Let me know if you need anything else.")
    elif stage == "get_name":
        data["Patient_Name"] = user_input
        st.session_state.booking_stage = "get_cnic"
        bot("Please enter your Patient_ID :")
    elif stage == "get_cnic":
        data["Patient_ID"] = user_input
        st.session_state.booking_stage = "get_phone"
        bot("Please enter your phone number:")
    elif stage == "get_phone":
        data["Phone No."] = user_input
        st.session_state.booking_stage = "get_history"
        bot("Briefly describe your issue or treatment needed:")
    elif stage == "get_history":
        data["History"] = user_input
        st.session_state.booking_stage = "choose_doctor"
        # Build a single, numbered list of doctors
        doctor_list = "\n".join(
            f"{i+1}. {name} — {info['specialty']}"
            for i, (name, info) in enumerate(DOCTOR_SCHEDULES.items())
        )


        # Send one consolidated message
        bot(
            "Which doctor would you like to see? Here are our available specialists:\n\n"
            f"{doctor_list}\n\n"
            "Please type the number of your chosen doctor:"
        )

    
    elif stage == "choose_doctor":
        try:
            idx = int(user_input.strip()) - 1
            doctors = list(DOCTOR_SCHEDULES.keys())
            if 0 <= idx < len(doctors):
                chosen = doctors[idx]
                data["Doctor"] = chosen
                st.session_state.booking_stage = "choose_slot"
                slots = get_slots_for_doctor(chosen)
                data["AvailableSlots"] = slots
                # Build a single, numbered list of slots
                slots_list = "\n".join(f"{i+1}. {slot}" for i, slot in enumerate(slots))

                # Send one consolidated message
                bot(
                    f"Available slots for {chosen}:\n\n"
                    f"{slots_list}\n\n"
                    "Please type the number of your preferred slot:"
                )
            else:
                bot("Invalid doctor number. Please choose from the list.")
        except:
            bot("Enter a valid number for the doctor (e.g., 1, 2).")

    elif stage == "choose_slot":
        # grab the list you saved earlier
        slots = data["AvailableSlots"]
        try:
            choice = int(user_input.strip()) - 1
            if 0 <= choice < len(slots):
                data["Appointment Date-Time"] = slots[choice]
                # move on to confirmation
                st.session_state.booking_stage = "confirm"

                # ask for final confirmation in one message
                bot(
                    f"Please confirm your appointment details:\n\n"
                    f"• **Patient Name:** {data['Patient_Name']}\n"
                    f"• **Patient ID:** {data['Patient_ID']}\n"
                    f"• **Phone No.:** {data['Phone No.']}\n"
                    f"• **History:** {data['History']}\n"
                    f"• **Doctor:** {data['Doctor']}\n"
                    f"• **Appointment Date-Time:** {data['Appointment Date-Time']}\n\n"
                    "Reply **1** to finalize or **skip** to cancel."
                )

            else:
                bot("That slot number isn’t valid—please choose a number from the list above.")
        except ValueError:
            bot("Please type a valid slot number (e.g. 1 or 2).")
        return





    elif stage == "confirm":
        if user_input.lower() == "1":
            save_appointment({
                "Patient_Name": data["Patient_Name"],
                "Patient_ID": data["Patient_ID"],
                "Phone No.": data["Phone No."],
                "History": data["History"],
                "Doctor": data["Doctor"],
                "Appointment Date-Time": data["Appointment Date-Time"],
                "Comment":"payment not done",
                "Status":"pending"
            })
            st.session_state.booking_stage = "none"
            st.session_state.appointment_data = {}
            bot("✅ Your appointment has been confirmed and saved. Thank you!")
        else:
            st.session_state.booking_stage = "none"
            st.session_state.appointment_data = {}
            bot("Appointment booking has been cancelled. Let me know if you need help with anything else.")

if __name__ == "__main__":
    main()
