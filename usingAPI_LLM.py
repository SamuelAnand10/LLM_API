import streamlit as st
from gradio_client import Client

# --- Configuration and Aesthetics ---

# Set up the page layout and title for an aesthetic look
st.set_page_config(
    page_title="AI Prompt Generator",
    page_icon="✨",
    layout="centered", # Centers the main content on wide screens
    initial_sidebar_state="collapsed",
)

# Apply custom CSS for better visual appeal
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #1c1c1c;
    }
    /* Style the main "Generate" button */
    .stButton>button {
        background-color: #4CAF50; /* Primary green */
        color: white;
        border-radius: 12px;
        font-weight: bold;
        padding: 10px 24px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    /* Style for text area and containers */
    .stTextArea textarea, .stContainer {
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Core Logic ---

# Use st.cache_resource to initialize the Gradio Client only once.
# This significantly speeds up the app.
@st.cache_resource
def get_gradio_client():
    """Initializes and returns the Gradio Client."""
    CLIENT_URL = "https://f4b1bb5c13d8313f42.gradio.live/"
    st.info(f"Connecting to Gradio endpoint at: `{CLIENT_URL}`")
    try:
        client = Client(CLIENT_URL)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Gradio Client. Please check the URL. Error: {e}")
        return None

client = get_gradio_client()

def get_prediction(client, prompt: str):
    """
    Calls the Gradio API with the fixed parameters from the user's original script.
    """
    API_NAME = "/lambda"
    
    try:
        # Use st.spinner to show a dynamic loading state while fetching data
        with st.spinner(f"Sending prompt and awaiting response..."):
            result = client.predict(
                q=prompt,
                mt=170,    # Max Tokens
                t=0.1,     # Temperature
                p=0.95,    # Top P
                api_name=API_NAME
            )
            return result
    except Exception as e:
        # Catch and display network or API errors gracefully
        st.error(f"An error occurred during API call: {e}")
        return None

# --- Application UI ---

st.title("✨ AI Text Generation App")
st.markdown("Powered by **Streamlit** and a **Gradio** hosted model.")

if client is None:
    st.error("The Gradio client could not be initialized. Check console for details.")

else:
    # Text input for the user's prompt
    prompt = st.text_area(
        "**Enter Your Prompt**",
        placeholder="e.g., Describe a futuristic city powered entirely by renewable energy.",
        height=180
    )

    st.markdown("---")
    
    # Button to trigger the generation
    if st.button("Generate Response", type="primary", use_container_width=True):
        if prompt:
            # Clear previous results if any
            st.session_state['result'] = get_prediction(client, prompt)
            st.session_state['current_prompt'] = prompt
        else:
            st.warning("Please enter a prompt to begin generation.")

    # Display Results only if a prediction has been made
    if 'result' in st.session_state and st.session_state['result'] is not None:
        st.subheader("Model Output")
        
        # Use a container with a border for a professional look
        with st.container(border=True):
            st.markdown("### 📝 Your Query")
            st.info(st.session_state['current_prompt'])
            
            st.markdown("### 🤖 Response")
            st.success(st.session_state['result'])

        # Optional details footer
        st.caption(f"Used API: `/lambda` | Max Tokens: 128 | Temp: 0.7 | Top P: 0.95")

st.divider()
st.markdown(
    "<div style='text-align: center; color: #777; font-size: small;'>Built with Streamlit & Gradio Client</div>",
    unsafe_allow_html=True
)
